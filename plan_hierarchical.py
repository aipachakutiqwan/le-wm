"""Hierarchical LeWM inference script.

Loads a stage-2 HierarchicalLeWM checkpoint and evaluates it in an environment
using the two-level CEM-MPC planner.  Follows the same structure as eval.py.

Usage
-----
python plan_hierarchical.py checkpoint=<path/to/hierarchical_lewm_object.ckpt>

# Different number of eval episodes
python plan_hierarchical.py checkpoint=<path> eval.num_eval=10

# Run on GPU
python plan_hierarchical.py checkpoint=<path> device=cuda
"""

import copy
import gc
import os
import logging
import threading
import time
from collections import deque
from pathlib import Path

py_log = logging.getLogger(__name__)

os.environ["MUJOCO_GL"] = "egl"

import hydra
import numpy as np
import torch
import stable_worldmodel as swm
import stable_pretraining as spt
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms


from hierarchical_plan import plan, compile_for_planning


# ──────────────────────────────────────────────────────────────────────────────
# Thread worker for 2-GPU environment split
# ──────────────────────────────────────────────────────────────────────────────


def _plan_env_worker(model, z_init_slice, z_goal_slice, plan_kwargs, stats, results, idx):
    """Run plan() sequentially over a slice of environments; store result or exception."""
    try:
        actions = []
        for i in range(z_init_slice.shape[0]):
            a = plan(model, z_init_slice[i], z_goal_slice[i], **plan_kwargs, stats=stats)
            actions.append(a.cpu().numpy())
        results[idx] = np.stack(actions)   # (slice_envs, action_dim)
    except Exception as e:
        results[idx] = e


# ──────────────────────────────────────────────────────────────────────────────
# Policy
# ──────────────────────────────────────────────────────────────────────────────


class HierarchicalPolicy(swm.policy.BasePolicy):
    """MPC policy backed by the two-level CEM planner.

    Inherits _prepare_info (image transforms + column normalisation) from
    BasePolicy, then encodes the preprocessed observations and calls plan_batched()
    for all environments in one shot.

    When extra_device is provided a second copy of the model is loaded onto that
    device and environments are split across both GPUs in parallel threads,
    roughly halving wall-clock planning time.

    Parameters
    ----------
    model          : trained HierarchicalLeWM
    plan_cfg       : OmegaConf node with H_high / h_low / *_samples / *_iters
    process        : dict of sklearn-style column normalisers (same as eval.py)
    transform      : dict of torchvision image transforms keyed by obs key
    device         : primary torch device string (e.g. "cuda:0")
    extra_device   : optional second device string (e.g. "cuda:1"); when set a
                     deepcopy of the model is placed there and environments are
                     split evenly between the two GPUs via threading.
    compile_planner: call compile_for_planning() on each device's model.
                     Adds a ~2 min one-time warm-up; saves ~20 % per call after.
    """

    def __init__(
        self,
        model,
        plan_cfg,
        process,
        transform,
        device,
        extra_device: str | None = None,
        compile_planner: bool = False,
        log_every: int = 5,
        eval_budget: int | None = None,
    ):
        super().__init__()
        if compile_planner:
            compile_for_planning(model)
        self.model = model.eval().to(device)
        self.plan_cfg = plan_cfg
        self.process = process
        self.transform = transform
        self.device = device
        self._action_queue: deque = deque()
        self._frameskip: int | None = None
        self._plan_step: int = 0
        self._n_plan_total: int | None = None
        self._eval_budget: int | None = eval_budget
        self._plan_stats: dict = {}
        self._log_every: int = log_every
        self._t_start: float = time.time()

        # Optional second GPU.
        self._extra_device: str | None = None
        self._extra_model = None
        if extra_device is not None:
            py_log.info("Replicating model to %s for two-GPU planning", extra_device)
            extra_model = copy.deepcopy(model).eval().to(extra_device)
            if compile_planner:
                compile_for_planning(extra_model)
            self._extra_model = extra_model
            self._extra_device = extra_device

    def set_env(self, env) -> None:
        self.env = env
        self._action_queue.clear()
        self._plan_step = 0
        self._n_plan_total = None
        self._plan_stats = {}
        self._t_start = time.time()

    def _encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode pixel tensor to latent states.

        Parameters
        ----------
        pixels : (E, T, C, H, W)  — E environments, T timesteps

        Returns
        -------
        (E, D)  — last-timestep latent for each environment
        """
        pixels = pixels.to(self.device)
        with torch.no_grad():
            emb = self.model.jepa.encode({"pixels": pixels})["emb"]
        return emb[:, -1]   # (E, D)

    def get_action(self, info_dict: dict, **kwargs) -> np.ndarray:
        """Plan and return the next primitive action for each environment.

        All environments are planned in one batched call to plan_batched().
        When a second GPU is configured, environments are split evenly across
        both devices and the two halves run concurrently in threads.

        Parameters
        ----------
        info_dict : raw observation dict from swm.World (pixels, goal, …)

        Returns
        -------
        (num_envs, base_action_dim) numpy array, denormalised
        """
        if self._action_queue:
            return self._action_queue.popleft()

        info_dict = self._prepare_info(info_dict)
        z_init = self._encode(info_dict["pixels"])   # (E, D)
        z_goal = self._encode(info_dict["goal"])     # (E, D)
        n_envs = z_init.shape[0]
        self._plan_step += 1

        # Track mean latent L1 distance across all envs (keys distinct from plan()'s own
        # per-env "init_dist0"/"prev_dist" keys, which would corrupt these mean values).
        dist_now = (z_goal - z_init).abs().sum(-1).mean().item()
        if "emb_init" not in self._plan_stats:
            self._plan_stats["emb_init"] = dist_now
        self._plan_stats["emb_prev"] = dist_now

        plan_kwargs = dict(
            H_high=self.plan_cfg.H_high,
            h_low=self.plan_cfg.h_low,
            outer_samples=self.plan_cfg.outer_samples,
            inner_samples=self.plan_cfg.inner_samples,
            outer_iters=self.plan_cfg.outer_iters,
            inner_iters=self.plan_cfg.inner_iters,
            outer_std=self.plan_cfg.get("outer_std", 5.0),
            inner_std=self.plan_cfg.get("inner_std", 1.0),
            step=self._plan_step,
        )

        if self._extra_model is not None and n_envs >= 2:
            # Split environments between two GPUs and run each half sequentially per-env.
            # Sequential plan() keeps peak VRAM at 1 env × outer_samples, avoiding the
            # GPU saturation that kills MuJoCo's EGL contexts with batched planning.
            # Thread 0 owns self._plan_stats to avoid races; thread 1 discards stats.
            split = n_envs // 2
            results = [None, None]
            threads = [
                threading.Thread(
                    target=_plan_env_worker,
                    args=(
                        self.model,
                        z_init[:split].to(self.device),
                        z_goal[:split].to(self.device),
                        plan_kwargs,
                        self._plan_stats,
                        results, 0,
                    ),
                ),
                threading.Thread(
                    target=_plan_env_worker,
                    args=(
                        self._extra_model,
                        z_init[split:].to(self._extra_device),
                        z_goal[split:].to(self._extra_device),
                        plan_kwargs,
                        None,
                        results, 1,
                    ),
                ),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            for i, r in enumerate(results):
                if isinstance(r, Exception):
                    raise RuntimeError(f"GPU thread {i} failed") from r
            eff = np.concatenate([results[0], results[1]], axis=0)
        else:
            actions = []
            for i in range(n_envs):
                a = plan(
                    self.model, z_init[i], z_goal[i],
                    **plan_kwargs, stats=self._plan_stats,
                )
                actions.append(a.cpu().numpy())
            eff = np.stack(actions)

        if "action" in self.process:
            scaler = self.process["action"]
            base_dim = scaler.n_features_in_
            if self._frameskip is None:
                self._frameskip = eff.shape[-1] // base_dim
            fs = self._frameskip
            # eff[e] is [prim_0, prim_1, ..., prim_{fs-1}] concatenated. Split per env
            # first (E, fs, base), inverse-transform per row, then transpose to
            # (fs, E, base) so prim[t, e] is env e's t-th primitive. Reshaping straight
            # from (E*fs, base) to (fs, E, base) scrambles primitives across envs.
            prim = eff.reshape(n_envs, fs, base_dim)
            prim = scaler.inverse_transform(prim.reshape(-1, base_dim))
            prim = prim.reshape(n_envs, fs, base_dim).transpose(1, 0, 2)
        else:
            base_dim = eff.shape[-1]
            if self._frameskip is None:
                self._frameskip = 1
            fs = self._frameskip
            prim = eff.reshape(fs, n_envs, base_dim)

        # queue steps 1..fs-1; return step 0 immediately
        for t in range(1, fs):
            self._action_queue.append(prim[t])   # (E, base_dim)

        # Compute total plan calls once frameskip is known.
        if self._n_plan_total is None and self._eval_budget is not None:
            self._n_plan_total = max(1, self._eval_budget // self._frameskip)

        if self._plan_step % self._log_every == 0:
            elapsed = time.time() - self._t_start
            st = self._plan_stats
            avg_s = st.get("total_ms", 0.0) / self._plan_step / 1000.0
            total_str = f"/{self._n_plan_total}" if self._n_plan_total is not None else ""
            dist_str = ""
            if "emb_init" in st and "emb_prev" in st:
                dist_str = (
                    f"  emb_dist {float(st['emb_init']):.3f}"
                    f"→{float(st['emb_prev']):.3f}"
                )
            py_log.info(
                "plan_step %3d%s  elapsed %5.0fs  avg_plan %.1fs/call%s",
                self._plan_step, total_str, elapsed, avg_s, dist_str,
            )

        return prim[0]   # (E, base_dim)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (mirror eval.py)
# ──────────────────────────────────────────────────────────────────────────────


def img_transform(img_size: int):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=img_size),
    ])


def get_dataset(cfg):
    dataset_path = Path(cfg.get("cache_dir") or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        cfg.dataset.name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    return np.array([np.max(step_idx[episode_idx == ep]) + 1 for ep in episodes])


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


@hydra.main(version_base=None, config_path="./config/eval", config_name="hierarchical_tworoom")
def run(cfg: DictConfig):
    py_log.info("Hierarchical eval — checkpoint=%s device=%s", cfg.checkpoint, cfg.device)

    assert cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget, (
        "plan_config.horizon * action_block must be <= eval.eval_budget"
    )

    ##########################
    ##     environment      ##
    ##########################

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(cfg.eval.img_size, cfg.eval.img_size))

    ##########################
    ##      dataset         ##
    ##########################

    dataset = get_dataset(cfg)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        scaler = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        scaler.fit(col_data)
        process[col] = scaler
        if col != "action":
            process[f"goal_{col}"] = scaler

    transform = {
        "pixels": img_transform(cfg.eval.img_size),
        "goal": img_transform(cfg.eval.img_size),
    }

    ##########################
    ##       model          ##
    ##########################

    py_log.info("Loading checkpoint from %s", cfg.checkpoint)
    model = torch.load(cfg.checkpoint, map_location=cfg.device, weights_only=False)

    if cfg.get("random_policy", False):
        py_log.info("DIAGNOSTIC: using RandomPolicy (planner disabled)")
        policy = swm.policy.RandomPolicy()
    else:
        policy = HierarchicalPolicy(
            model=model,
            plan_cfg=cfg.plan,
            process=process,
            transform=transform,
            device=cfg.device,
            extra_device=cfg.get("extra_device", None),
            compile_planner=cfg.get("compile_planner", False),
            eval_budget=cfg.eval.eval_budget,
        )

    ##########################
    ##     episode sample   ##
    ##########################

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_dict = {ep: max_start[i] for i, ep in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_dict[ep] for ep in dataset.get_col_data(col_name)]
    )
    valid_idx = np.nonzero(dataset.get_col_data("step_idx") <= max_start_per_row)[0]
    py_log.info("%d valid starting points found", len(valid_idx))

    rng = np.random.default_rng(cfg.seed)
    chosen = np.sort(
        valid_idx[rng.choice(len(valid_idx), size=cfg.eval.num_eval, replace=False)]
    )

    eval_episodes = dataset.get_row_data(chosen)[col_name]
    eval_start_idx = dataset.get_row_data(chosen)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    # Initial agent-to-goal distance per episode (tworoom diagnostic).
    # Uses pos_agent when available; silently skipped for envs that don't expose it.
    _dist_col = cfg.eval.get("dist_col", "pos_agent")
    if _dist_col in dataset.column_names:
        start_pos = np.asarray(dataset.get_row_data(chosen)[_dist_col])
        goal_pos = np.asarray(dataset.get_row_data(chosen + cfg.eval.goal_offset_steps)[_dist_col])
        init_dist = np.linalg.norm(start_pos - goal_pos, axis=-1)
    else:
        init_dist = None

    ##########################
    ##      evaluation      ##
    ##########################

    world.set_policy(policy)
    results_path = Path(cfg.checkpoint).parent

    py_log.info(
        "Evaluating %d episodes × %d-step budget  "
        "[H=%d h=%d oi=%d ii=%d outer_std=%.1f inner_std=%.1f]",
        cfg.eval.num_eval, cfg.eval.eval_budget,
        cfg.plan.H_high, cfg.plan.h_low,
        cfg.plan.outer_iters, cfg.plan.inner_iters,
        cfg.plan.get("outer_std", 5.0), cfg.plan.get("inner_std", 1.0),
    )

    t0 = time.time()
    try:
        metrics = world.evaluate_from_dataset(
            dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset_steps=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
            video_path=results_path,
        )
    finally:
        world.close()   # free EGL contexts
        del world
        gc.collect()    # destroy GL Python objects now, before EGL display terminates at shutdown
    elapsed = time.time() - t0

    py_log.info("metrics: %s", metrics)
    py_log.info("evaluation time: %.1f s", elapsed)

    # Per-episode breakdown: does success correlate with starting near the goal?
    # If successes are concentrated at small init_dist, the 20% is "free" (the planner
    # isn't earning it) — points to a model/execution problem rather than weak search.
    succ = np.asarray(metrics.get("episode_successes"))
    if init_dist is not None and succ is not None and succ.shape == init_dist.shape:
        order = np.argsort(init_dist)
        py_log.info("per-episode (sorted by initial distance to goal):")
        for j in order:
            py_log.info("  ep=%-5s init_dist=%.3f  success=%s",
                        int(eval_episodes[j]), float(init_dist[j]), bool(succ[j]))
        if succ.any():
            py_log.info("mean init_dist | success=%.3f  fail=%.3f",
                        float(init_dist[succ].mean()),
                        float(init_dist[~succ].mean()) if (~succ).any() else float("nan"))

    sr = float(np.mean(succ)) if (succ is not None and succ.ndim > 0) else float("nan")
    env_tag = cfg.world.env_name.replace("/", "_")
    ckpt_path = Path(cfg.checkpoint)
    auto_name = (
        f"{env_tag}_H{cfg.plan.H_high}_h{cfg.plan.h_low}"
        f"_oi{cfg.plan.outer_iters}_ii{cfg.plan.inner_iters}"
        f"_n{cfg.eval.num_eval}_seed{cfg.seed}_sr{sr:.3f}"
        f"_{ckpt_path.parent.name}_{ckpt_path.stem}.txt"
    )
    out = results_path / (cfg.output.get("filename") or auto_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write("\n==== MODEL ====\n")
        f.write(f"checkpoint_folder: {ckpt_path.parent}\n")
        f.write(f"checkpoint_name:   {ckpt_path.name}\n")
        f.write("\n==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {elapsed:.1f} s\n")

    py_log.info("Results written to %s", out)


if __name__ == "__main__":
    run()
