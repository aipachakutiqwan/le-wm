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

import os
import logging
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


from hierarchical_plan import plan


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

    def __init__(self, model, plan_cfg, process, transform, device):
        super().__init__()
        self.model = model.eval().to(device)
        self.plan_cfg = plan_cfg
        self.process = process
        self.transform = transform
        self.device = device
        self._action_queue: deque = deque()
        # effective_action_dim = frameskip * base_action_dim; derived at first get_action call
        self._frameskip: int | None = None

    def set_env(self, env) -> None:
        self.env = env
        self._action_queue.clear()

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
        if self._action_queue:
            return self._action_queue.popleft()

        info_dict = self._prepare_info(info_dict)

        # after _prepare_info: pixels / goal are (E, T, C, H, W) tensors
        z_init = self._encode(info_dict["pixels"])   # (E, D)
        z_goal = self._encode(info_dict["goal"])     # (E, D)

        n_envs = z_init.shape[0]
        effective_actions = []
        for i in range(n_envs):
            a = plan(
                self.model,
                z_init[i],
                z_goal[i],
                H_high=self.plan_cfg.H_high,
                h_low=self.plan_cfg.h_low,
                outer_samples=self.plan_cfg.outer_samples,
                inner_samples=self.plan_cfg.inner_samples,
                outer_iters=self.plan_cfg.outer_iters,
                inner_iters=self.plan_cfg.inner_iters,
            )
            effective_actions.append(a.cpu().numpy())

        # effective_actions: list of (effective_action_dim,) → stack to (E, eff_dim)
        eff = np.stack(effective_actions)

        if "action" in self.process:
            scaler = self.process["action"]
            base_dim = scaler.n_features_in_
            if self._frameskip is None:
                self._frameskip = eff.shape[-1] // base_dim
            fs = self._frameskip
            # reshape to (E * fs, base_dim), inverse-transform, reshape to (fs, E, base_dim)
            prim = eff.reshape(n_envs * fs, base_dim)
            prim = scaler.inverse_transform(prim)
            prim = prim.reshape(fs, n_envs, base_dim)
        else:
            base_dim = eff.shape[-1]
            if self._frameskip is None:
                self._frameskip = 1
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

    policy = HierarchicalPolicy(
        model=model,
        plan_cfg=cfg.plan,
        process=process,
        transform=transform,
        device=cfg.device,
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
    py_log.info("evaluation time: %.1f s", elapsed)

    out = results_path / cfg.output.filename
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write("\n==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {elapsed:.1f} s\n")

    py_log.info("Results written to %s", out)


if __name__ == "__main__":
    run()
