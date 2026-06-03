"""Script A — static encoder cost field vs P^(2) best-reachable cost.

This is design #1 "as stated". It is NOT a strict apples-to-apples comparison:
  * top row  = STATIC, optimisation-free latent distance  ||E(state) - E(goal)||
               (this is the same kind of field as the original cost_landscape figure)
  * bottom row = best cost the HIGH-LEVEL planner can reach after the outer CEM
               min_l ||P^(2)_rollout(E(state), l) - E(goal)||
The bottom row bakes in an optimisation the top row does not, so the two rows are
different *types* of object. For the symmetric (truly 1-to-1) version, where both
rows are best-reachable fields, see landscape_reachability_pair.py.

Usage
-----
STABLEWM_HOME=$HOME/.stable_worldmodel \
    .venv/bin/python diagnostics/landscape_static_vs_highlevel.py --device cuda
"""
import argparse
from pathlib import Path

import torch

from _landscape_lib import (get_jepa, encode_frames, sample_frames, static_field,
                            reach_field, render_grid, GOALS_XY, goal_index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/hierarchical_lewm_epoch_14_tworooms_object.ckpt")
    ap.add_argument("--dataset", default="tworoom")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-starts", type=int, default=1200)
    ap.add_argument("--metric", choices=["l1", "l2"], default="l1")
    ap.add_argument("--high-iters", type=int, default=5)
    ap.add_argument("--high-samples", type=int, default=256)
    ap.add_argument("--high-hhigh", type=int, default=1)      # H_high; tuned=1 (stale default was 4)
    ap.add_argument("--outer-std", type=float, default=2.5)   # macro CEM std; tuned=2.5 (stale was 5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "landscape_static_vs_highlevel.png"))
    args = ap.parse_args()

    model = torch.load(args.checkpoint, map_location=args.device, weights_only=False).eval()
    jepa = get_jepa(model)
    pixels, pos = sample_frames(args.dataset, args.n_starts, args.seed)
    Z = encode_frames(jepa, pixels, args.device)
    print(f"loaded {args.checkpoint}; encoder {type(jepa).__name__}; starts={len(Z)}")

    static_rows, high_rows = [], []
    for gxy in GOALS_XY:
        zg = Z[goal_index(pos, gxy)]
        print(f"goal {gxy}: static + high-reach ...")
        static_rows.append(static_field(Z, zg, args.metric))
        high_rows.append(reach_field(model, Z, zg, level="high", device=args.device,
                                     metric=args.metric, steps=args.high_hhigh,
                                     std0=args.outer_std, iters=args.high_iters,
                                     samples=args.high_samples))

    render_grid(
        pos,
        rows=[("STATIC  ||E(s)-E(g)||  (no planning)", static_rows),
              ("P(2) best-reachable  min_l ||P2(s,l)-E(g)||", high_rows)],
        out=args.out,
        suptitle=f"TwoRoom: static encoder cost vs P(2) best-reachable "
                 f"({args.metric.upper()}, N={args.n_starts}, H_high={args.high_hhigh}, outer_std={args.outer_std})",
        metric=args.metric,
    )


if __name__ == "__main__":
    main()
