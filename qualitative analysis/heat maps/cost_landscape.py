"""Latent cost-landscape heatmap for TwoRoom.

For a fixed goal image, encode many frames spanning the arena with the frozen
LeWM encoder and plot latent distance(z, z_goal) as a heatmap over the agent's
true (x, y). This visualises the cost surface the CEM-MPC planner descends.

Two companion diagnostics make the reading rigorous:
  * a linear probe z -> (x, y): confirms position IS encoded (so weak cost
    contrast is a metric-geometry effect, not an encoding bug);
  * latent cost vs. straight-line distance: shows whether the cost gives a
    usable gradient everywhere or only in a local basin near the goal.

Offline: no environment, no planning loop — just encoder forward passes.

Usage
-----
STABLEWM_HOME=$HOME/.stable_worldmodel \
    .venv/bin/python "qualitative analysis/heat maps/cost_landscape.py" \
    --checkpoint baseline/tworoom/hierarchical_lewm_object.ckpt --device cuda
"""
import argparse
import sys
from pathlib import Path

# Repo root (holds jepa.py) must be importable so torch.load can unpickle model classes.
_root = next((p for p in Path(__file__).resolve().parents if (p / "jepa.py").exists()), None)
if _root is not None:
    sys.path.insert(0, str(_root))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import stable_worldmodel as swm

# ImageNet normalisation — matches img_transform() in plan_hierarchical.py / eval.py
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# TwoRoom arena: pos in [14, 209]^2; central vertical wall at the midline.
WALL_X = 111.5


def get_jepa(obj):
    """Return the inner JEPA (has .encode) from either a Stage-1 or H-LeWM ckpt."""
    if hasattr(obj, "jepa"):
        return obj.jepa
    if hasattr(obj, "encode"):
        return obj
    for m in obj.modules():
        if m is not obj and hasattr(m, "encode"):
            return m
    raise RuntimeError("no submodule with .encode found in checkpoint")


@torch.no_grad()
def encode_frames(jepa, pixels_hwc, device, batch=256):
    """pixels_hwc: (N,224,224,3) uint8  ->  latents (N, D)."""
    out = []
    for i in range(0, len(pixels_hwc), batch):
        px = torch.from_numpy(pixels_hwc[i:i + batch]).permute(0, 3, 1, 2).float() / 255.0
        px = ((px - _MEAN) / _STD).unsqueeze(1).to(device)   # (b, 1, 3, H, W)
        emb = jepa.encode({"pixels": px})["emb"][:, -1]       # (b, D)
        out.append(emb.float().cpu())
    return torch.cat(out).numpy()


def linear_probe_r2(Z, pos, seed=0):
    """Held-out R^2 of a linear map z -> (x, y). High R^2 => position is encoded."""
    n = len(Z)
    perm = np.random.default_rng(seed).permutation(n)
    ntr = int(0.8 * n)
    tr, te = perm[:ntr], perm[ntr:]
    A = np.concatenate([Z, np.ones((n, 1))], axis=1)
    W, *_ = np.linalg.lstsq(A[tr], pos[tr], rcond=None)
    pred = A[te] @ W
    ss_res = ((pos[te] - pred) ** 2).sum(0)
    ss_tot = ((pos[te] - pos[te].mean(0)) ** 2).sum(0)
    return 1 - ss_res / ss_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="baseline/tworoom/hierarchical_lewm_object.ckpt")
    ap.add_argument("--dataset", default="tworoom")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-frames", type=int, default=6000)
    ap.add_argument("--metric", choices=["l2", "l1"], default="l2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "cost_landscape_tworoom.png"))
    args = ap.parse_args()

    model = torch.load(args.checkpoint, map_location=args.device, weights_only=False).eval()
    jepa = get_jepa(model)
    print(f"loaded {args.checkpoint} -> encoder {type(jepa).__name__}")

    ds = swm.data.HDF5Dataset(args.dataset, keys_to_cache=["proprio"])
    rng = np.random.default_rng(args.seed)
    rows = np.sort(rng.choice(len(ds), size=args.n_frames, replace=False))
    rd = ds.get_row_data(rows)
    pixels = np.asarray(rd["pixels"])                  # (N,224,224,3) uint8
    pos = np.asarray(rd["pos_agent"]).astype(float)    # (N,2)
    print(f"frames={pixels.shape}  pos extent min={pos.min(0).round(1)} max={pos.max(0).round(1)}")

    Z = encode_frames(jepa, pixels, args.device)       # (N, D)
    print(f"latents={Z.shape}")

    # --- sanity: is position linearly decodable from the latent? ---
    r2 = linear_probe_r2(Z, pos, seed=args.seed)
    print(f"linear probe z->(x,y) held-out R^2:  x={r2[0]:.3f}  y={r2[1]:.3f}  mean={r2.mean():.3f}")

    def cost_to(zg):
        d = Z - zg[None]
        return np.abs(d).sum(1) if args.metric == "l1" else (d ** 2).sum(1)

    # Goals chosen to expose the cross-wall story: one in each room + a far corner.
    goals_xy = [(55, 110), (165, 110), (185, 185)]
    titles = ["Goal in LEFT room", "Goal in RIGHT room", "Goal in RIGHT corner"]

    # ---- Figure 1: spatial cost heatmaps ----
    fig, axes = plt.subplots(1, len(goals_xy), figsize=(5.2 * len(goals_xy), 4.8),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    print(f"\n--- cost summary ({args.metric.upper()}) ---")
    for ax, gxy, title in zip(axes, goals_xy, titles):
        gi = int(np.argmin(((pos - np.array(gxy)) ** 2).sum(1)))
        c = cost_to(Z[gi])
        # Compress the far plateau so the local basin near the goal is legible.
        vmin, vmax = c.min(), np.percentile(c, 65)
        hb = ax.hexbin(pos[:, 0], pos[:, 1], C=c, gridsize=30, cmap="RdYlGn_r",
                       reduce_C_function=np.mean, vmin=vmin, vmax=vmax)
        ax.axvline(WALL_X, color="k", lw=1.0, ls="--", alpha=0.5)
        ax.plot(pos[gi, 0], pos[gi, 1], marker="*", ms=22, mfc="white", mec="black", mew=1.5)
        ax.set_title(f"{title}  (goal ≈ {pos[gi].round(0)})", fontsize=11)
        ax.set_xlabel("x position", fontsize=10)
        ax.set_ylabel("y position", fontsize=10)
        ax.set_aspect("equal")
        ax.invert_yaxis()  # match rendered image (small y = top)
        cb = fig.colorbar(hb, ax=ax, shrink=0.85)
        cb.set_label(f"latent {args.metric.upper()} dist to goal\n(green = CLOSER · red = farther)", fontsize=9)

        same = (pos[:, 0] < WALL_X) == (gxy[0] < WALL_X)
        eucl = np.sqrt(((pos - pos[gi]) ** 2).sum(1))
        r = np.corrcoef(c, eucl)[0, 1]
        print(f"{title:22s} corr(latent,straight-line)={r:+.2f}  "
              f"same-room={c[same].mean():.1f}  opp-room={c[~same].mean():.1f}  "
              f"ratio={c[~same].mean() / c[same].mean():.2f}")

    fig.suptitle(f"LeWM latent cost landscape on TwoRoom "
                 f"({args.metric.upper()} latent distance, N={args.n_frames} frames)", fontsize=13)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"saved {args.out}")

    # ---- Figure 2: latent cost vs. true straight-line distance ----
    out2 = str(Path(args.out).with_name(Path(args.out).stem + "_vs_distance.png"))
    fig2, ax2 = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    for gxy, title in zip(goals_xy, titles):
        gi = int(np.argmin(((pos - np.array(gxy)) ** 2).sum(1)))
        c = cost_to(Z[gi])
        dist = np.sqrt(((pos - pos[gi]) ** 2).sum(1))
        bins = np.linspace(0, dist.max(), 16)
        idx = np.digitize(dist, bins)
        bx = [dist[idx == b].mean() for b in range(1, len(bins)) if (idx == b).any()]
        by = [c[idx == b].mean() for b in range(1, len(bins)) if (idx == b).any()]
        ax2.plot(bx, by, marker="o", ms=4, label=title.replace("Goal in ", ""))
    ax2.set_xlabel("straight-line distance to goal (arena units)", fontsize=10)
    ax2.set_ylabel(f"mean latent {args.metric.upper()} distance to goal", fontsize=10)
    ax2.set_title("Latent cost vs. true distance\n(rises only locally, then saturates = weak long-range signal)",
                  fontsize=11)
    ax2.legend(fontsize=9)
    fig2.savefig(out2, dpi=140)
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
