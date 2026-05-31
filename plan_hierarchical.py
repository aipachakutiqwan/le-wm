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
import os
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
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

from hierarchical_plan import plan, plan_batched


# ──────────────────────────────────────────────────────────────────────────────
# Policy
# ──────────────────────────────────────────────────────────────────────────────


class HierarchicalPolicy(swm.policy.BasePolicy):
    """MPC policy backed by the two-level CEM planner.

    Inherits _prepare_info (image transforms + column normalisation) from
    BasePolicy, then encodes the preprocessed observations and calls plan().

    Parameters
    ----------
    model      : trained HierarchicalLeWM
    plan_cfg   : OmegaConf node with H_high / h_low / *_samples / *_iters
    process    : dict of sklearn-style column normalisers (same as eval.py)
    transform  : dict of torchvision image transforms keyed by obs key
    device     : torch device string
    """

    def __init__(self, model, plan_cfg, process, transform, device, eval_budget: int = 0):
        super().__init__()
        self.model = model.eval().to(device)
        self.plan_cfg = plan_cfg
        self.process = process
        self.transform = transform
        self.device = device
        self._eval_budget = eval_budget          # total env steps per episode (for X/Y display)
        self._action_queue: deque = deque()
        # effective_action_dim = frameskip * base_action_dim; derived at first get_action call
        self._frameskip: int | None = None
        self._total_plan_steps: int | None = None  # derived once frameskip is known
        self._env_step: int = 0
        self._plan_step: int = 0
        self._total_plan_time: float = 0.0         # cumulative seconds spent in plan()
        # Outer-CEM warm-start: keyed by "mu_mac" (E, H_high, d_L) on CPU.
        # Updated in-place by plan_batched after each call; cleared on episode reset.
        self._warmstart: dict = {}

        # Build per-GPU model replicas for multi-GPU planning.
        # Each entry is (device_str, model_replica); single GPU / CPU → one entry.
        n_cuda = torch.cuda.device_count()
        if n_cuda > 1:
            self._gpu_replicas = [
                (f"cuda:{i}", copy.deepcopy(self.model).eval().to(f"cuda:{i}"))
                for i in range(n_cuda)
            ]
            py_log.info("Multi-GPU planning: %d GPUs — environments will be sharded across them", n_cuda)
        else:
            self._gpu_replicas = [(device, self.model)]

        # Compile rollout loops per replica, after deepcopy.
        # mode="default" (Inductor): unrolls the small fixed loops and fuses ops.
        # mode="reduce-overhead" (CUDA graphs) is intentionally NOT used: the
        # rolling torch.cat inside _rollout_high/_rollout_low grows z_seq each
        # step, producing dynamic intermediate shapes that CUDA graphs can't capture.
        n_compiled = 0
        for dev, replica in self._gpu_replicas:
            if dev.startswith("cuda"):
                replica._rollout_high = torch.compile(replica._rollout_high, mode="default")
                replica._rollout_low  = torch.compile(replica._rollout_low,  mode="default")
                n_compiled += 1
        if n_compiled:
            py_log.info("Compiled _rollout_high and _rollout_low on %d GPU replica(s)", n_compiled)

    def set_env(self, env) -> None:
        self.env = env
        self._action_queue.clear()
        self._warmstart.clear()
        self._env_step = 0
        self._plan_step = 0
        self._total_plan_time = 0.0

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

    def _plan_multi_gpu(
        self,
        z_init: torch.Tensor,
        z_goal: torch.Tensor,
        plan_kwargs: dict,
    ) -> np.ndarray:
        """Shard environments across GPUs and run plan_batched on each shard in parallel."""
        n_gpus = len(self._gpu_replicas)
        # tensor_split handles uneven E gracefully (last shard may be 1 smaller)
        z_init_shards = torch.tensor_split(z_init, n_gpus, dim=0)
        z_goal_shards = torch.tensor_split(z_goal, n_gpus, dim=0)

        # Shard the outer-CEM warm-start along the env axis (same split as z_init).
        mu_mac_full = self._warmstart.get("mu_mac")
        mu_mac_shards = (
            torch.tensor_split(mu_mac_full, n_gpus, dim=0)
            if mu_mac_full is not None
            else [None] * n_gpus
        )

        def _run_shard(dev, replica, zi, zg, mu_i):
            ws: dict = {} if mu_i is None else {"mu_mac": mu_i.to(dev)}
            action = plan_batched(replica, zi.to(dev), zg.to(dev), warmstart=ws, **plan_kwargs)
            # ws["mu_mac"] is now the updated best_mac for this shard (on dev).
            updated_mu = ws["mu_mac"].cpu() if "mu_mac" in ws else None
            return action.cpu().numpy(), updated_mu

        # skip empty shards (can occur when E < n_gpus)
        active = [
            (dr, zi, zg, mu)
            for dr, zi, zg, mu in zip(self._gpu_replicas, z_init_shards, z_goal_shards, mu_mac_shards)
            if zi.shape[0] > 0
        ]
        with ThreadPoolExecutor(max_workers=len(active)) as pool:
            futures = [
                pool.submit(_run_shard, dev, replica, zi, zg, mu)
                for (dev, replica), zi, zg, mu in active
            ]
            results = [f.result() for f in futures]

        # Merge updated warm-starts back into self._warmstart.
        new_mus = [r[1] for r in results if r[1] is not None]
        if new_mus:
            self._warmstart["mu_mac"] = torch.cat(new_mus, dim=0)

        return np.concatenate([r[0] for r in results], axis=0)

    def get_action(self, info_dict: dict, **kwargs) -> np.ndarray:
        """Plan and return the next primitive action for each environment.

        plan() returns an *effective* action of shape (frameskip * base_dim,).
        We split it into frameskip primitive actions, inverse-transform each,
        and serve them one per call via an internal queue so the world only
        needs frame_skip=1.

        Parameters
        ----------
        info_dict : raw observation dict from swm.World (pixels, goal, …)

        Returns
        -------
        (num_envs, base_action_dim) numpy array, denormalised
        """
        self._env_step += 1
        env_prog = f"{self._env_step}/{self._eval_budget}" if self._eval_budget else str(self._env_step)
        if self._action_queue:
            py_log.debug("env_step=%s  serving queued action (%d remaining)", env_prog, len(self._action_queue))
            return self._action_queue.popleft()

        self._plan_step += 1
        plan_prog = f"{self._plan_step}/{self._total_plan_steps}" if self._total_plan_steps else str(self._plan_step)
        py_log.info("env_step=%s  plan_step=%s  running CEM planner", env_prog, plan_prog)

        info_dict = self._prepare_info(info_dict)

        # after _prepare_info: pixels / goal are (E, T, C, H, W) tensors
        z_init = self._encode(info_dict["pixels"])   # (E, D)
        z_goal = self._encode(info_dict["goal"])     # (E, D)
        n_envs = z_init.shape[0]

        plan_kwargs = dict(
            H_high=self.plan_cfg.H_high,
            h_low=self.plan_cfg.h_low,
            outer_samples=self.plan_cfg.outer_samples,
            inner_samples=self.plan_cfg.inner_samples,
            outer_iters=self.plan_cfg.outer_iters,
            inner_iters=self.plan_cfg.inner_iters,
        )
        _t0 = time.perf_counter()
        if len(self._gpu_replicas) > 1:
            eff = self._plan_multi_gpu(z_init, z_goal, plan_kwargs)
        else:
            eff = plan_batched(
                self.model, z_init, z_goal, warmstart=self._warmstart, **plan_kwargs
            ).cpu().numpy()
        _plan_elapsed = time.perf_counter() - _t0
        self._total_plan_time += _plan_elapsed
        py_log.info("  plan done in %.2f s  (total planning time: %.1f s)", _plan_elapsed, self._total_plan_time)

        if "action" in self.process:
            scaler = self.process["action"]
            base_dim = scaler.n_features_in_
            if self._frameskip is None:
                self._frameskip = eff.shape[-1] // base_dim
                if self._eval_budget:
                    self._total_plan_steps = max(1, self._eval_budget // self._frameskip)
            fs = self._frameskip
            # eff is (E, fs*base_dim); split env-outer then swap to (fs, E, base_dim).
            # A flat reshape (E*fs, base_dim) → (fs, E, base_dim) is wrong in C-order
            # because it treats the first axis as timestep-outer instead of env-outer.
            prim = scaler.inverse_transform(eff.reshape(n_envs * fs, base_dim))  # (E*fs, base_dim)
            prim = prim.reshape(n_envs, fs, base_dim).swapaxes(0, 1)             # (fs, E, base_dim)
        else:
            base_dim = eff.shape[-1]
            if self._frameskip is None:
                self._frameskip = 1
                if self._eval_budget:
                    self._total_plan_steps = self._eval_budget
            fs = self._frameskip
            prim = eff.reshape(fs, n_envs, base_dim)

        # queue steps 1..fs-1; return step 0 immediately
        for t in range(1, fs):
            self._action_queue.append(prim[t])   # (E, base_dim)

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
    t_run = time.perf_counter()
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
    # Unwrap any torch.compile wrappers saved into the checkpoint.
    # OptimizedModules cannot be deepcopy'd across threads (_plan_multi_gpu)
    # and trigger FX tracing errors when called from a ThreadPoolExecutor.
    model.action_encoder_high = getattr(model.action_encoder_high, '_orig_mod', model.action_encoder_high)
    model.high_predictor = getattr(model.high_predictor, '_orig_mod', model.high_predictor)

    policy = HierarchicalPolicy(
        model=model,
        plan_cfg=cfg.plan,
        process=process,
        transform=transform,
        device=cfg.device,
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
        valid_idx[rng.choice(len(valid_idx) - 1, size=cfg.eval.num_eval, replace=False)]
    )

    eval_episodes = dataset.get_row_data(chosen)[col_name]
    eval_start_idx = dataset.get_row_data(chosen)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    ##########################
    ##      evaluation      ##
    ##########################

    world.set_policy(policy)
    results_path = Path(cfg.checkpoint).parent

    t0 = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video_path=results_path,
    )
    elapsed = time.time() - t0

    py_log.info("metrics: %s", metrics)
    py_log.info("evaluation time:      %.1f s (%.1f min)", elapsed, elapsed / 60)
    py_log.info("total planning time:  %.1f s (%.1f%% of eval)", policy._total_plan_time, 100 * policy._total_plan_time / elapsed)

    sr = metrics.get("success_rate", metrics.get("success", float("nan")))
    env_slug = cfg.dataset.name.replace("/", "_")
    p = cfg.plan
    out_name = (
        f"{env_slug}"
        f"_sr{sr:.2f}"
        f"_H{p.H_high}_h{p.h_low}"
        f"_oS{p.outer_samples}_oi{p.outer_iters}"
        f"_iS{p.inner_samples}_ii{p.inner_iters}"
        f"_n{cfg.eval.num_eval}_bgt{cfg.eval.eval_budget}"
        f".txt"
    )
    out = results_path / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write("\n==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {elapsed:.1f} s\n")

    py_log.info("Results written to %s", out)
    total_s = time.perf_counter() - t_run
    py_log.info("run complete — total time: %.1f s (%.1f min)", total_s, total_s / 60)


if __name__ == "__main__":
    run()
