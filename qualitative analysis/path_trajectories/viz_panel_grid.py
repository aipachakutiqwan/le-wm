"""2x2 montage: one hand-picked episode per horizon, flat vs hier overlaid.

Reuses draw_layout + load from viz_trajectories.py (same folder, so it is on
sys.path[0] when this file is run as a script). Edit PANELS to change which
(horizon, episode) lands in each cell.

    python "qualitative analysis/path_trajectories/viz_panel_grid.py" \
        --out "qualitative analysis/path_trajectories/runs/figures/horizon_2x2.png"
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from viz_trajectories import draw_layout, load

RUN = "qualitative analysis/path_trajectories/runs"   # relative to repo-root cwd

# (cell label, horizon dir suffix, episode id) — row-major
PANELS = [
    ("d=25",  "d25",  4339),
    ("d=50",  "d50",  7756),
    ("d=75",  "d75",  886),
    ("d=100", "d100", 5017),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{RUN}/figures/horizon_2x2.png")
    a = ap.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(7, 7.8), squeeze=False)
    for ax, (label, d, ep) in zip(axes.flat, PANELS):
        flat = load(f"{RUN}/flat_{d}/trajectories_flat.npz")
        hier = load(f"{RUN}/hier_{d}/trajectories_hier.npz")
        draw_layout(ax)
        tags = []
        for src, color, name in [(flat, "tab:blue", "flat"), (hier, "tab:orange", "hier")]:
            if ep not in src:
                continue
            e = src[ep]
            ls = "-" if e["success"] else "--"        # solid = success, dashed = failure
            ax.plot(e["path"][:, 0], e["path"][:, 1], ls, color=color, lw=1.8)
            tags.append(f"{name} {'✓' if e['success'] else '✗'}")
        ref = flat.get(ep) or hier.get(ep)            # start/goal identical across planners
        ax.scatter(*ref["start"], c="k", marker="o", s=45, zorder=5)
        ax.scatter(*ref["goal"], c="k", marker="*", s=160, zorder=5)
        ax.set_xlabel(f"{label}  ·  ep {ep}   ({'  '.join(tags)})", fontsize=10)

    legend = [
        Line2D([], [], color="tab:blue", lw=2, label="flat LeWM"),
        Line2D([], [], color="tab:orange", lw=2, label="H-LeWM"),
        Line2D([], [], color="k", marker="o", ls="", label="start"),
        Line2D([], [], color="k", marker="*", ls="", label="goal"),
        Line2D([], [], color="0.4", lw=2, label="solid = success  ·  dashed = failure"),
    ]
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=5, fontsize=9, frameon=False)
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
