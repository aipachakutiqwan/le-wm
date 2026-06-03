"""Paper Fig. 9-style latent visualization (Tier-2: clean state grid).

Builds a near-uniform grid of TwoRoom states from the dataset (bin the x-y plane,
keep the frame nearest each cell center), encodes those representatives with the
frozen JEPA encoder, and projects to 2-D with t-SNE. Each point is colored by a
bilinear 4-corner 2D colormap of its (x, y), and the same colors are reused in
both panels:

    left  : physical (x, y) grid colored by the 2D map   (doubles as the legend)
    right : t-SNE latent embedding, colored by the SAME per-point colors

The grid (vs. random frames) removes density artifacts, so the latent panel
shows the smoothly-deformed "folded sheet" of LeWM's Fig. 9 when topology is
preserved. No env / MuJoCo needed — states come from real dataset frames.

This is additive: latent_tsne.py / latent_tsne_2d.py and their outputs are
untouched.

Usage
-----
STABLEWM_HOME=/home/kaboo/.stable_worldmodel \
python diagnostics/latent_grid_tsne.py \
    --checkpoint /home/kaboo/.stable_worldmodel/20260527_004340/hierarchical_lewm_object.ckpt \
    --device cuda
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (latent_analysis -> qualitative analysis -> le-wm); unpickle HierarchicalLeWM

import matplotlib
matplotlib.use("Agg")  # headless WSL
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import stable_worldmodel as swm

# ImageNet normalisation — must match img_transform() in plan_hierarchical.py
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# 2D colormap corners (mimics the paper's blue/red/gray/green scheme)
_BL = np.array([0.20, 0.30, 0.85])  # low x, low y  -> blue
_BR = np.array([0.85, 0.20, 0.20])  # high x, low y -> red
_TL = np.array([0.62, 0.62, 0.62])  # low x, high y -> gray
_TR = np.array([0.20, 0.65, 0.30])  # high x, high y-> green


def make_2d_colors(pos):
    x, y = pos[:, 0], pos[:, 1]
    u = ((x - x.min()) / (np.ptp(x) + 1e-9))[:, None]
    v = ((y - y.min()) / (np.ptp(y) + 1e-9))[:, None]
    rgb = (1 - u) * (1 - v) * _BL + u * (1 - v) * _BR \
        + (1 - u) * v * _TL + u * v * _TR
    return np.clip(rgb, 0.0, 1.0)


def collect_frames(dataset, num_windows, seed):
    """Load `num_windows` windows; return all frames (uint8) + agent (x,y)."""
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(dataset), size=min(num_windows, len(dataset)), replace=False)
    frames, pos = [], []
    for i in idxs:
        s = dataset[int(i)]
        frames.append(s["pixels"])              # (num_steps, 3, H, W) uint8
        pos.append(np.asarray(s["proprio"]))    # (num_steps, 2)
    frames = torch.cat(frames, dim=0)
    pos = np.concatenate(pos, axis=0)
    keep = ~np.isnan(pos).any(axis=1)
    return frames[keep], pos[keep]


def grid_representatives(pos, grid):
    """Bin (x,y) into grid x grid cells; return index of the frame nearest each
    occupied cell's center -> one representative per reachable cell."""
    x, y = pos[:, 0], pos[:, 1]
    xmin, ymin, xspan, yspan = x.min(), y.min(), np.ptp(x) + 1e-9, np.ptp(y) + 1e-9
    gx = np.clip(((x - xmin) / xspan * grid).astype(int), 0, grid - 1)
    gy = np.clip(((y - ymin) / yspan * grid).astype(int), 0, grid - 1)
    cell = gx * grid + gy
    reps = []
    for c in np.unique(cell):
        members = np.where(cell == c)[0]
        cx = ((c // grid) + 0.5) / grid * xspan + xmin
        cy = ((c % grid) + 0.5) / grid * yspan + ymin
        d = (x[members] - cx) ** 2 + (y[members] - cy) ** 2
        reps.append(members[d.argmin()])
    return np.array(reps)


@torch.no_grad()
def encode_frames(model, frames, device, batch=128):
    mean, std = _MEAN.to(device), _STD.to(device)
    out = []
    for k in range(0, len(frames), batch):
        chunk = frames[k:k + batch].to(device).float() / 255.0
        chunk = (chunk - mean) / std
        emb = model.jepa.encode({"pixels": chunk.unsqueeze(1)})["emb"]
        out.append(emb[:, 0].cpu())
    return torch.cat(out, dim=0).numpy()


def project(Z, perplexity, seed):
    n_pca = min(50, Z.shape[1], Z.shape[0])
    Z_pca = PCA(n_components=n_pca, random_state=seed).fit_transform(Z)
    perp = min(perplexity, max(5, len(Z) // 4))
    Z_tsne = TSNE(n_components=2, perplexity=perp, init="pca",
                  learning_rate="auto", random_state=seed).fit_transform(Z_pca)
    return Z_tsne, perp


def plot(pos, tsne, rgb, out, n, perp):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5))
    axL.scatter(pos[:, 0], pos[:, 1], c=rgb, s=22, alpha=0.95, edgecolors="none")
    axL.set_title(f"TwoRoom physical state grid (N={n}) — color legend", fontsize=13)
    axL.set_xlabel("agent x", fontsize=11)
    axL.set_ylabel("agent y", fontsize=11)
    axL.set_aspect("equal", adjustable="datalim")

    axR.scatter(tsne[:, 0], tsne[:, 1], c=rgb, s=22, alpha=0.95, edgecolors="none")
    axR.set_title("2D projection (latent space, t-SNE)", fontsize=13)
    axR.set_xlabel("t-SNE 1", fontsize=11)
    axR.set_ylabel("t-SNE 2", fontsize=11)

    for ax in (axL, axR):
        ax.tick_params(labelsize=9)
    fig.suptitle(
        f"TwoRoom LeWM latent space — frozen ViT-tiny encoder, grid (perplexity={perp})\n"
        "smooth color flow in the latent panel => spatial topology preserved",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved figure -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", default="tworoom")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-windows", type=int, default=700)
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--grid", type=int, default=24, help="grid resolution (cells per axis)")
    ap.add_argument("--perplexity", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "figures" / "latent_grid_tworoom.png"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"checkpoint={args.checkpoint}  device={args.device}")
    model = torch.load(args.checkpoint, map_location=args.device, weights_only=False).eval()
    dataset = swm.data.HDF5Dataset(
        name=args.dataset, num_steps=args.num_steps, frameskip=args.frameskip,
        keys_to_load=["pixels", "action", "proprio"],
        keys_to_cache=["action", "proprio"], transform=None,
    )

    frames, pos = collect_frames(dataset, args.num_windows, args.seed)
    print(f"collected {len(frames)} frames")
    reps = grid_representatives(pos, args.grid)
    rep_frames, rep_pos = frames[reps], pos[reps]
    print(f"grid {args.grid}x{args.grid}: {len(reps)} occupied cells (representative frames)")

    Z = encode_frames(model, rep_frames, args.device)
    print(f"latents: shape={Z.shape}  finite={np.isfinite(Z).all()}")

    tsne, perp = project(Z, args.perplexity, args.seed)
    rgb = make_2d_colors(rep_pos)
    plot(rep_pos, tsne, rgb, args.out, len(reps), perp)

    npz = str(Path(args.out).with_suffix(".npz"))
    np.savez(npz, Z=Z, tsne=tsne, pos=rep_pos)
    print(f"saved arrays -> {npz}")


if __name__ == "__main__":
    main()
