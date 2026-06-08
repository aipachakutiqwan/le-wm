"""Plot TwoRoom rollout paths: flat vs. hierarchical, success vs. failure.

Reads the .npz files written by eval.py / plan_hierarchical.py
(`record_trajectories=true`) and renders one panel per shared episode.
No swm dependency — numpy + matplotlib only.

    python viz_trajectories.py --flat trajectories_flat.npz \
                               --hier trajectories_hier.npz --out paths.png
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ──────────────────────────────────────────────────────────────────────────
# TwoRoom layout — fixed for all episodes (envs/two_room/env.py constants)
# ──────────────────────────────────────────────────────────────────────────
IMG, BORDER = 224, 14            # canvas size + outer margin (valid area [14, 210])
WALL_X, WALL_HALF = 112, 5       # vertical wall at x=112, thickness 10
DOOR_Y, DOOR_HALF = 49, 14       # door centred at y=49 → open gap y∈[35, 63]


def draw_layout(ax):
    """Draw the arena border, the vertical wall, and the doorway gap."""
    lo, hi = BORDER, IMG - BORDER
    ax.add_patch(plt.Rectangle((lo, lo), hi - lo, hi - lo, fill=False, ec="0.6", lw=1))
    for y0, y1 in [(lo, DOOR_Y - DOOR_HALF), (DOOR_Y + DOOR_HALF, hi)]:   # wall = 2 segments
        ax.add_patch(plt.Rectangle((WALL_X - WALL_HALF, y0), 2 * WALL_HALF, y1 - y0, color="0.4"))
    ax.set_xlim(lo - 4, hi + 4)
    ax.set_ylim(lo - 4, hi + 4)
    ax.set_aspect("equal")
    ax.invert_yaxis()            # proprio y is top-down (image rows) → match the videos
    ax.set_xticks([])
    ax.set_yticks([])


# ──────────────────────────────────────────────────────────────────────────
# Data loading — npz → {episode_id: {path, success, start, goal}}
# ──────────────────────────────────────────────────────────────────────────
def load(path):
    if not path:
        return {}
    d = np.load(path, allow_pickle=True)
    pos, succ = d["positions"], d["episode_successes"]
    eps, start, goal = d["eval_episodes"], d["start_proprio"], d["goal_proprio"]
    return {
        int(ep): dict(path=pos[:, j], success=bool(succ[j]), start=start[j], goal=goal[j])
        for j, ep in enumerate(eps)
    }


# ──────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat", help="trajectories_flat.npz")
    ap.add_argument("--hier", help="trajectories_hier.npz")
    ap.add_argument("--out", default="trajectories.png")
    ap.add_argument("--max", type=int, default=12, help="max episodes to plot")
    ap.add_argument("--ncols", type=int, default=4)
    a = ap.parse_args()

    flat, hier = load(a.flat), load(a.hier)
    episodes = sorted(set(flat) | set(hier))[: a.max]
    n = len(episodes)
    ncols = min(a.ncols, n)
    nrows = -(-n // ncols)                        # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.2 * nrows), squeeze=False)

    for ax, ep in zip(axes.flat, episodes):
        draw_layout(ax)
        tags = []
        for src, color, name in [(flat, "tab:blue", "flat"), (hier, "tab:orange", "hier")]:
            if ep not in src:
                continue
            e = src[ep]
            ls = "-" if e["success"] else "--"    # solid = success, dashed = failure
            ax.plot(e["path"][:, 0], e["path"][:, 1], ls, color=color, lw=1.6)
            tags.append(f"{name} {'✓' if e['success'] else '✗'}")
        ref = flat.get(ep) or hier.get(ep)        # start/goal identical across planners
        ax.scatter(*ref["start"], c="k", marker="o", s=40, zorder=5)
        ax.scatter(*ref["goal"], c="k", marker="*", s=140, zorder=5)
        ax.set_title(f"ep {ep}  |  " + "  ".join(tags), fontsize=9)

    for ax in axes.flat[n:]:                       # blank any unused panels
        ax.axis("off")

    legend = [
        Line2D([], [], color="tab:blue", lw=2, label="flat LeWM"),
        Line2D([], [], color="tab:orange", lw=2, label="H-LeWM"),
        Line2D([], [], color="k", marker="o", ls="", label="start"),
        Line2D([], [], color="k", marker="*", ls="", label="goal"),
        Line2D([], [], color="0.4", lw=2, label="solid = success  ·  dashed = failure"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"wrote {a.out}  ({n} episodes)")


if __name__ == "__main__":
    main()
