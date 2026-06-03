"""Script B — the most 1-to-1 comparison of flat vs hierarchical planning.

Both rows render the SAME quantity: from each start position, the best terminal
latent distance to the goal the planner can achieve in one plan (min over its
actions). Only the planner changes:
  * top row  = FLAT       min_a ||P^(1)_rollout(E(s), a) - E(g)||   (primitive actions, h_low steps)
  * bottom row = HIERARCHICAL  min_l ||P^(2)_rollout(E(s), l) - E(g)||  (macro-actions, H_high steps)
Everything else (start frames, goals, metric, CEM, rendering) is identical, so the
difference between the rows is exactly the contribution of the hierarchical extension.

This is a per-PLAN reachability snapshot (one outer/inner plan), not a full MPC
episode — the right unit for comparing the two planners' cost surfaces.

Usage
-----
STABLEWM_HOME=$HOME/.stable_worldmodel \
    .venv/bin/python diagnostics/landscape_reachability_pair.py --device cuda
"""
import argparse
from pathlib import Path

import torch

from _landscape_lib import (get_jepa, encode_frames, sample_frames,
                            reach_field, render_grid, GOALS_XY, goal_index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/hierarchical_lewm_epoch_14_tworooms_object.ckpt")
    ap.add_argument("--dataset", default="tworoom")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-starts", type=int, default=1000)
    ap.add_argument("--metric", choices=["l1", "l2"], default="l1")
    # flat (P^1) inner CEM budget; the inner_iters lever was decisive in our results.
    ap.add_argument("--low-iters", type=int, default=20)
    ap.add_argument("--low-samples", type=int, default=128)
    ap.add_argument("--low-hlow", type=int, default=10)
    # hierarchical (P^2) outer CEM budget
    ap.add_argument("--high-iters", type=int, default=5)
    ap.add_argument("--high-samples", type=int, default=256)
    ap.add_argument("--high-hhigh", type=int, default=1)      # H_high; tuned=1 (stale default was 4)
    ap.add_argument("--outer-std", type=float, default=2.5)   # macro CEM std; tuned=2.5 (stale was 5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "landscape_reachability_pair.png"))
    args = ap.parse_args()

    model = torch.load(args.checkpoint, map_location=args.device, weights_only=False).eval()
    jepa = get_jepa(model)
    pixels, pos = sample_frames(args.dataset, args.n_starts, args.seed)
    Z = encode_frames(jepa, pixels, args.device)
    print(f"loaded {args.checkpoint}; encoder {type(jepa).__name__}; starts={len(Z)}")

    flat_rows, high_rows = [], []
    for gxy in GOALS_XY:
        zg = Z[goal_index(pos, gxy)]
        print(f"goal {gxy}: flat P1 reach ...")
        flat_rows.append(reach_field(model, Z, zg, level="low", device=args.device,
                                     metric=args.metric, steps=args.low_hlow,
                                     iters=args.low_iters, samples=args.low_samples))
        print(f"goal {gxy}: hierarchical P2 reach ...")
        high_rows.append(reach_field(model, Z, zg, level="high", device=args.device,
                                     metric=args.metric, steps=args.high_hhigh,
                                     std0=args.outer_std, iters=args.high_iters,
                                     samples=args.high_samples))

    render_grid(
        pos,
        rows=[("FLAT  P1 reachable  min_a ||P1(s,a)-E(g)||", flat_rows),
              ("HIER  P2 reachable  min_l ||P2(s,l)-E(g)||", high_rows)],
        out=args.out,
        suptitle=f"TwoRoom: best per-plan reachability, flat P(1) vs hierarchical P(2) "
                 f"({args.metric.upper()}, N={args.n_starts}, h_low={args.low_hlow}, H_high={args.high_hhigh}, outer_std={args.outer_std})",
        metric=args.metric,
    )


if __name__ == "__main__":
    main()
