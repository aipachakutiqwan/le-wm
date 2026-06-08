"""Macro-action latent space (A_psi) visualization — the hierarchy's OWN latent.

Unlike the encoder t-SNE (which maps the frozen, shared E latent space), this maps
the one new latent space the hierarchy introduces: the d_L=8 macro-action space
produced by the action encoder A_psi. For many real inter-waypoint action chunks
we compute the macro-action l = A_psi(chunk) and color each by the chunk's net
agent displacement (dx, dy) via the same bilinear 2D colormap:

    left  : net displacement (dx, dy) of each chunk, 2D-colored   (the legend)
    right : t-SNE of the 8-d macro-actions, colored by the SAME (dx, dy)

Coherent color flow in the right panel => A_psi organizes macro-actions by
direction/magnitude of motion (it learned a meaningful action abstraction).
Mixed color => the KL->N(0,I) regularizer washed motion structure out.

Actions are z-scored with the training StandardScaler before A_psi (feeding raw
actions is off-distribution — see hierarchical_probe.py's correction note).

Additive: the other diagnostics scripts and their outputs are untouched.

Usage
-----
STABLEWM_HOME=/home/kaboo/.stable_worldmodel \
python "analysis/latent_analysis/macro_action_tsne.py" \
    --checkpoint /home/kaboo/.stable_worldmodel/20260527_004340/hierarchical_lewm_object.ckpt \
    --device cuda
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (latent_analysis -> analysis -> le-wm); unpickle + waypoint_sampler

import matplotlib
matplotlib.use("Agg")  # headless WSL
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import preprocessing

import stable_worldmodel as swm
from utils.waypoint_sampler import sample_waypoints_fixed_stride

# 2D colormap corners (same scheme as the position figures)
_BL = np.array([0.20, 0.30, 0.85])  # low dx, low dy
_BR = np.array([0.85, 0.20, 0.20])  # high dx, low dy
_TL = np.array([0.62, 0.62, 0.62])  # low dx, high dy
_TR = np.array([0.20, 0.65, 0.30])  # high dx, high dy


def make_2d_colors(v2):
    a, b = v2[:, 0], v2[:, 1]
    u = ((a - a.min()) / (np.ptp(a) + 1e-9))[:, None]
    w = ((b - b.min()) / (np.ptp(b) + 1e-9))[:, None]
    rgb = (1 - u) * (1 - w) * _BL + u * (1 - w) * _BR \
        + (1 - u) * w * _TL + u * w * _TR
    return np.clip(rgb, 0.0, 1.0)


@torch.no_grad()
def macro_actions(model, act, wp_idx):
    """A_psi on each inter-waypoint chunk -> (M, n_seg, d_L). Matches the probe."""
    segs = []
    for k in range(len(wp_idx) - 1):
        chunk = torch.nan_to_num(act[:, wp_idx[k]:wp_idx[k + 1]], 0.0)
        segs.append(model.action_encoder_high(chunk))
    return torch.stack(segs, dim=1)


def collect(model, dataset, scaler, wp_idx, num_windows, max_points, seed, device):
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(dataset), size=min(num_windows, len(dataset)), replace=False)
    acts, props = [], []
    for i in idxs:
        s = dataset[int(i)]
        acts.append(s["action"])                 # (T, A_eff) torch
        props.append(np.asarray(s["proprio"]))   # (T, 2)
    act = torch.stack(acts).float()              # (M, T, A_eff)
    prop = np.stack(props)                        # (M, T, 2)

    # z-score actions exactly like training (scaler fit on base_dim raw actions)
    M, T, A_eff = act.shape
    base = scaler.n_features_in_
    arr = act.numpy().reshape(M, T, A_eff // base, base)
    arr = (arr - scaler.mean_) / scaler.scale_
    act = torch.from_numpy(arr.reshape(M, T, A_eff)).float().to(device)

    macros = macro_actions(model, act, wp_idx)    # (M, n_seg, d_L)
    disp = np.stack([prop[:, wp_idx[k + 1]] - prop[:, wp_idx[k]]
                     for k in range(len(wp_idx) - 1)], axis=1)  # (M, n_seg, 2)

    macros = macros.reshape(-1, macros.shape[-1]).cpu().numpy()
    disp = disp.reshape(-1, 2)
    keep = ~np.isnan(disp).any(axis=1) & np.isfinite(macros).all(axis=1)
    macros, disp = macros[keep], disp[keep]
    if len(macros) > max_points:
        sel = rng.choice(len(macros), size=max_points, replace=False)
        macros, disp = macros[sel], disp[sel]
    return macros, disp


def plot(disp, tsne, rgb, out, n, perp):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5))
    axL.scatter(disp[:, 0], disp[:, 1], c=rgb, s=10, alpha=0.85, edgecolors="none")
    axL.set_title(r"chunk net displacement $(\Delta x, \Delta y)$ — color legend", fontsize=13)
    axL.set_xlabel(r"$\Delta x$ over chunk", fontsize=11)
    axL.set_ylabel(r"$\Delta y$ over chunk", fontsize=11)
    axL.axhline(0, color="k", lw=0.5, alpha=0.3)
    axL.axvline(0, color="k", lw=0.5, alpha=0.3)
    axL.set_aspect("equal", adjustable="datalim")

    axR.scatter(tsne[:, 0], tsne[:, 1], c=rgb, s=10, alpha=0.85, edgecolors="none")
    axR.set_title(r"$A_\psi$ macro-action space (t-SNE), colored by motion", fontsize=13)
    axR.set_xlabel("t-SNE 1", fontsize=11)
    axR.set_ylabel("t-SNE 2", fontsize=11)

    for ax in (axL, axR):
        ax.tick_params(labelsize=9)
    fig.suptitle(
        f"TwoRoom H-LeWM macro-action latent ($A_\\psi$, $d_L$=8), N={n} chunks, perplexity={perp}\n"
        "color flow in the t-SNE => macro-actions organize by direction/magnitude of motion",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved figure -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", default="tworoom")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-windows", type=int, default=2000)
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--n-waypoints", type=int, default=4)
    ap.add_argument("--max-points", type=int, default=5000)
    ap.add_argument("--perplexity", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "figures" / "macro_action_tworoom.png"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"checkpoint={args.checkpoint}  device={args.device}")
    model = torch.load(args.checkpoint, map_location=args.device, weights_only=False).eval()
    print(f"latent_action_dim (d_L) = {model.latent_action_dim}")

    # actions only — no pixels needed (fast, light)
    dataset = swm.data.HDF5Dataset(
        name=args.dataset, num_steps=args.num_steps, frameskip=args.frameskip,
        keys_to_load=["action", "proprio"],
        keys_to_cache=["action", "proprio"], transform=None,
    )
    flat = swm.data.HDF5Dataset(args.dataset, keys_to_cache=["action", "proprio"])
    acol = np.asarray(flat.get_col_data("action"))
    acol = acol[~np.isnan(acol).any(axis=1)]
    scaler = preprocessing.StandardScaler().fit(acol)

    wp_idx = sample_waypoints_fixed_stride(args.num_steps, N=args.n_waypoints).tolist()
    print(f"waypoint indices = {wp_idx}  ({len(wp_idx) - 1} chunks/window)")

    macros, disp = collect(model, dataset, scaler, wp_idx,
                           args.num_windows, args.max_points, args.seed, args.device)
    print(f"macros: shape={macros.shape}  "
          f"per-dim |mean|={np.abs(macros.mean(0)).mean():.3f}  std={macros.std(0).mean():.3f}  "
          f"(near 0 / 1 => KL->N(0,I) reg active)")
    print(f"displacement range: dx [{disp[:,0].min():.1f},{disp[:,0].max():.1f}]  "
          f"dy [{disp[:,1].min():.1f},{disp[:,1].max():.1f}]")

    perp = min(args.perplexity, max(5, len(macros) // 4))
    tsne = TSNE(n_components=2, perplexity=perp, init="pca",
                learning_rate="auto", random_state=args.seed).fit_transform(macros)
    rgb = make_2d_colors(disp)
    plot(disp, tsne, rgb, args.out, len(macros), perp)

    npz = str(Path(args.out).with_suffix(".npz"))
    np.savez(npz, macros=macros, tsne=tsne, disp=disp)
    print(f"saved arrays -> {npz}")


if __name__ == "__main__":
    main()
