"""Latent planning cost vs. goal offset (the eval's difficulty knob): Reacher vs TwoRoom.

How the eval uses the offset (eval.py): the goal is the state `goal_offset_steps`
ahead in the SAME episode (max_start = ep_len - offset - 1; goal = state[start+offset]).
So the planner's cost at the start is exactly  ||E(start) - E(start+Delta)||_2.

This plots that quantity directly as a function of goal offset Delta, for both
envs, together with the true physical start->goal distance ||pos[start+Delta] -
pos[start]||. Read: if the latent cost saturates with Delta while the physical
distance keeps growing, the planner can no longer tell a near goal from a far one
-> long-horizon collapse.

Offset is capped by episode length (Reacher 201 steps, TwoRoom <=101): no episode
is 500 long, so the feasible range is ~0-200 / ~0-100, not 0-500. --max-offset is
clamped to each env's longest episode.

Method: encode whole sampled episodes once, then every Delta is a vectorised
within-episode lookup. Offline (encoder forward passes only); standalone.

Usage
-----
STABLEWM_HOME=$HOME/.stable_worldmodel \
    .venv/bin/python "qualitative analysis/heat maps/latent_cost_vs_offset.py" --device cuda
"""
import argparse
import os
import sys
from pathlib import Path

_root = next((p for p in Path(__file__).resolve().parents if (p / "jepa.py").exists()), None)
if _root is not None:
    sys.path.insert(0, str(_root))                                # repo root for torch.load unpickle

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import stable_worldmodel as swm

# ImageNet normalisation — matches img_transform() in plan_hierarchical.py / eval.py
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


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
        px = ((px - _MEAN) / _STD).unsqueeze(1).to(device)
        emb = jepa.encode({"pixels": px})["emb"][:, -1]
        out.append(emb.float().cpu())
    return torch.cat(out).numpy()


EVAL_DELTAS = [25, 50, 75, 100]   # the offsets used in the success-rate sweep
CONFIGS = {
    "Reacher  (flat 78/94/88/80%)": dict(
        ckpt="baseline/reacher/lewm_epoch_10_object.ckpt", dataset="reacher",
        pos_key="finger_pos", units="sim units"),
    "TwoRoom  (flat 88/50/34/12%)": dict(
        ckpt="baseline/tworoom/lewm_epoch_9_object.ckpt", dataset="tworoom",
        pos_key="pos_agent", units="arena units"),
}


def encode_episodes(jepa, dataset, h5_path, pos_key, n_eps, device, seed):
    """Encode every frame of n_eps random episodes. Returns (list[(Z,P)], max_len)."""
    with h5py.File(h5_path, "r") as f:
        offs = np.asarray(f["ep_offset"][:])
        lens = np.asarray(f["ep_len"][:])
    rng = np.random.default_rng(seed)
    chosen = rng.permutation(len(offs))[:n_eps]
    eps = []
    for j, e in enumerate(chosen):
        o, L = int(offs[e]), int(lens[e])
        rd = dataset.get_row_data(np.arange(o, o + L))
        Z = encode_frames(jepa, np.asarray(rd["pixels"]), device)   # (L, D)
        P = np.asarray(rd[pos_key]).astype(float)                   # (L, 2)
        eps.append((Z, P))
        if (j + 1) % 25 == 0:
            print(f"    encoded {j + 1}/{n_eps} episodes")
    return eps, int(lens.max())


def cost_vs_offset(eps, deltas):
    """For each Delta: pooled within-episode latent L2 cost and physical distance."""
    cost, dist = {}, {}
    for dl in deltas:
        if dl == 0:
            cost[dl] = np.zeros(1)
            dist[dl] = np.zeros(1)
            continue
        cs, ps = [], []
        for Z, P in eps:
            if len(Z) > dl:
                cs.append(np.linalg.norm(Z[:-dl] - Z[dl:], axis=1))
                ps.append(np.linalg.norm(P[:-dl] - P[dl:], axis=1))
        cost[dl] = np.concatenate(cs) if cs else np.array([np.nan])
        dist[dl] = np.concatenate(ps) if ps else np.array([np.nan])
    return cost, dist


def summarize(d, deltas):
    """mean, q25, q75 arrays over the delta grid."""
    m = np.array([np.nanmean(d[dl]) for dl in deltas])
    lo = np.array([np.nanpercentile(d[dl], 25) for dl in deltas])
    hi = np.array([np.nanpercentile(d[dl], 75) for dl in deltas])
    return m, lo, hi


def norm01(m, lo, hi):
    """Min-max normalise the mean curve to [0,1]; scale the band the same way."""
    a, b = np.nanmin(m), np.nanmax(m)
    s = (b - a) or 1.0
    return (m - a) / s, (lo - a) / s, (hi - a) / s


def analyze(cfg, args):
    model = torch.load(cfg["ckpt"], map_location=args.device, weights_only=False).eval()
    jepa = get_jepa(model)
    ds = swm.data.HDF5Dataset(cfg["dataset"], keys_to_cache=[cfg["pos_key"]])
    stablewm = os.environ.get("STABLEWM_HOME") or os.path.expanduser("~/.stable_worldmodel")
    h5_path = os.path.join(stablewm, f"{cfg['dataset']}.h5")
    eps, max_len = encode_episodes(jepa, ds, h5_path, cfg["pos_key"],
                                   args.n_eps, args.device, args.seed)
    env_max = min(args.max_offset, max_len - 1)            # offset capped by episode length
    deltas = list(range(0, env_max + 1, args.step))
    cost, dist = cost_vs_offset(eps, deltas)
    return dict(deltas=np.array(deltas), cost=cost, dist=dist,
                env_max=env_max, max_len=max_len, units=cfg["units"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-eps", type=int, default=120)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--max-offset", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "latent_cost_vs_offset.png"))
    args = ap.parse_args()

    res = {}
    for name, cfg in CONFIGS.items():
        print(f"=== {name}  ({cfg['ckpt']}) ===")
        r = analyze(cfg, args)
        res[name] = r
        print(f"  episodes max_len={r['max_len']} -> offset capped at {r['env_max']} "
              f"(requested {args.max_offset})")
        for dl in EVAL_DELTAS:
            if dl in r["cost"]:
                print(f"  Delta={dl:3d}: latent_cost(mean)={np.nanmean(r['cost'][dl]):.1f}  "
                      f"phys_dist(mean)={np.nanmean(r['dist'][dl]):.3g} {r['units']}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), constrained_layout=True)
    for ax, (name, r) in zip(axes, res.items()):
        d = r["deltas"]
        cm, clo, chi = norm01(*summarize(r["cost"], d))
        dm, _, _ = norm01(*summarize(r["dist"], d))
        ax.plot(d, cm, "-o", ms=3, color="k", label="latent cost (norm.)")
        ax.fill_between(d, clo, chi, color="0.6", alpha=0.25, label="latent cost IQR")
        ax.plot(d, dm, "--", color="tab:blue", lw=2, label="true physical distance (norm.)")
        for dl in EVAL_DELTAS:
            if dl <= r["env_max"]:
                ax.axvline(dl, color="0.5", ls=":", lw=1.0)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("goal offset $\\Delta$ (timesteps)", fontsize=10)
        ax.set_xlim(0, r["env_max"])
        ax.set_ylim(-0.05, 1.1)
        ax.text(0.98, 0.04, f"offset capped at {r['env_max']}\n(episodes ≤ {r['max_len']} steps)",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="0.4")
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("min-max normalised (per env)", fontsize=10)
    fig.suptitle("Latent planning cost vs. goal offset $\\Delta$  "
                 "(dotted = eval offsets 25/50/75/100)", fontsize=12)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print("saved", args.out)


if __name__ == "__main__":
    main()
