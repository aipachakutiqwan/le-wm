"""Record per-step agent positions during TwoRoom eval (planner-agnostic).

Wraps any swm policy and logs ``info['proprio'][:, -1]`` (pixel-space x,y) each
step, so rollout paths can be plotted later (see ``viz_trajectories.py``).
Opt-in via the eval scripts' ``record_trajectories`` flag → default eval path
is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# ──────────────────────────────────────────────────────────────────────────
# Recording wrapper
# ──────────────────────────────────────────────────────────────────────────
class RecordingPolicy:
    """Transparent policy proxy that records the agent path.

    ``world.set_policy`` only needs ``set_env`` / ``get_action`` (+ optional
    ``seed``), so this delegates everything and just snoops the position.
    Works for flat, random, and hierarchical policies alike.
    """

    def __init__(self, inner, pos_key: str = "proprio"):
        self.inner = inner
        self.pos_key = pos_key          # TwoRoom: 'proprio' (NOT 'pos_agent', a dataset-only col)
        self.positions: list = []       # one (n_envs, 2) array per step

    def set_env(self, env):
        self.positions = []             # new episode batch
        return self.inner.set_env(env)

    def get_action(self, info, **kw):
        # Snapshot BEFORE delegating: inner._prepare_info() mutates & normalises
        # `info` in place, so we'd otherwise capture a normalised tensor.
        pos = np.asarray(info[self.pos_key])
        if pos.ndim == 3:               # (n_envs, history, 2) -> last timestep
            pos = pos[:, -1]
        self.positions.append(np.array(pos, copy=True))   # (n_envs, 2)
        return self.inner.get_action(info, **kw)

    def trajectories(self):
        """Stacked path of shape (T, n_envs, 2), or None if nothing recorded."""
        return np.stack(self.positions) if self.positions else None

    def __getattr__(self, name):        # proxy anything else (seed, type, ...) to inner
        inner = self.__dict__.get("inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)


# ──────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────
def save_trajectories_npz(path, policy, metrics, eval_episodes, eval_start_idx,
                          start_proprio, goal_proprio, **extra):
    """Dump paths + per-env metadata (all aligned by env index j) to an .npz."""
    traj = policy.trajectories()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        positions=traj if traj is not None else np.empty((0, 0, 2)),   # (T, envs, 2)
        episode_successes=np.asarray(metrics.get("episode_successes")),
        eval_episodes=np.asarray(eval_episodes),
        eval_start_idx=np.asarray(eval_start_idx),
        start_proprio=np.asarray(start_proprio),    # (envs, 2)
        goal_proprio=np.asarray(goal_proprio),      # (envs, 2)
        **extra,
    )
    return path
