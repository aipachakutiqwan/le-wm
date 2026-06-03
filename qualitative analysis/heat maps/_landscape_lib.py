"""Shared helpers for the TwoRoom latent cost-landscape figures.

Used by:
  landscape_static_vs_highlevel.py   static encoder cost  vs  P^(2) best-reachable
  landscape_reachability_pair.py     flat P^(1) reachable vs  hierarchical P^(2) reachable

Everything is offline (encoder + predictor forward passes only; no env / MuJoCo).
"""
import sys
from pathlib import Path

# Repo root (holds jepa.py / hierarchical_lewm.py) must be importable so torch.load
# can unpickle the saved model classes; locate it by walking up from this file.
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

WALL_X = 111.5  # TwoRoom central wall (pos in [14, 209]^2)
GOALS_XY = [(55, 110), (165, 110), (185, 185)]
GOAL_TITLES = ["goal: LEFT room", "goal: RIGHT room", "goal: RIGHT corner"]


def get_jepa(obj):
    """Return the inner JEPA (has .encode) from a Stage-1 or H-LeWM checkpoint."""
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
    """pixels_hwc: (N,224,224,3) uint8 -> latents (N, D)."""
    out = []
    for i in range(0, len(pixels_hwc), batch):
        px = torch.from_numpy(pixels_hwc[i:i + batch]).permute(0, 3, 1, 2).float() / 255.0
        px = ((px - _MEAN) / _STD).unsqueeze(1).to(device)
        out.append(jepa.encode({"pixels": px})["emb"][:, -1].float().cpu())
    return torch.cat(out).numpy()


def sample_frames(dataset_name, n, seed):
    """Return (pixels (n,224,224,3) uint8, pos (n,2)) for n random rows."""
    ds = swm.data.HDF5Dataset(dataset_name, keys_to_cache=["proprio"])
    rows = np.sort(np.random.default_rng(seed).choice(len(ds), size=n, replace=False))
    rd = ds.get_row_data(rows)
    return np.asarray(rd["pixels"]), np.asarray(rd["pos_agent"]).astype(float)


def goal_index(pos, gxy):
    return int(np.argmin(((pos - np.array(gxy)) ** 2).sum(1)))


def static_field(Z, z_goal, metric):
    """Static, optimisation-free latent distance from each frame to the goal."""
    d = Z - z_goal[None]
    return np.abs(d).sum(1) if metric == "l1" else (d ** 2).sum(1)


def _dist(zf, zg, metric):
    d = zf - zg
    return d.abs().sum(-1) if metric == "l1" else d.pow(2).sum(-1)


@torch.no_grad()
def reach_field(model, Z, z_goal, *, level, device, metric="l1",
                steps=None, std0=None, samples=None, iters=None,
                start_chunk=300, roll_chunk=2048, verbose=True):
    """Per-start best-achievable cost-to-goal via CEM, vectorised over start states.

    level="low"  : optimise primitive-action sequences, roll P^(1)  (flat planner)
    level="high" : optimise macro-action  sequences,    roll P^(2)  (hierarchical)

    Same algorithm as hierarchical_plan.cem (diagonal Gaussian, top-10% elites,
    std floor 0.1). Returns cost per start, shape (len(Z),).
    """
    M, D = Z.shape
    if level == "high":
        adim = model.latent_action_dim
        steps = steps or 4
        std0 = 5.0 if std0 is None else std0
        samples = samples or 256
        iters = iters or 5

        def rollout(z0, acts):
            return model._rollout_high(z0, acts)[:, -1]   # (B, D)
    elif level == "low":
        adim = model.action_dim
        steps = steps or 10
        std0 = 1.0 if std0 is None else std0
        samples = samples or 128
        iters = iters or 20

        def rollout(z0, acts):
            return model._rollout_low(z0, acts)           # (B, D)
    else:
        raise ValueError(level)

    Zt = torch.from_numpy(Z).float().to(device)
    zg = torch.from_numpy(np.asarray(z_goal)).float().to(device).view(1, D)
    n_elite = max(1, int(samples * 0.1))
    out = np.empty(M)

    for s0 in range(0, M, start_chunk):
        Zb = Zt[s0:s0 + start_chunk]               # (mb, D)
        mb = Zb.shape[0]
        mu = torch.zeros(mb, steps, adim, device=device)
        std = torch.full((mb, steps, adim), std0, device=device)
        for _ in range(iters):
            eps = torch.randn(mb, samples, steps, adim, device=device)
            cand = mu[:, None] + std[:, None] * eps        # (mb, S, steps, adim)
            flatc = cand.reshape(mb * samples, steps, adim)
            z0rep = Zb[:, None].expand(mb, samples, D).reshape(mb * samples, D)
            zf = torch.cat([rollout(z0rep[i:i + roll_chunk], flatc[i:i + roll_chunk])
                            for i in range(0, mb * samples, roll_chunk)])
            cost = _dist(zf, zg, metric).reshape(mb, samples)
            idx = cost.argsort(1)[:, :n_elite]             # (mb, n_elite)
            gi = idx[:, :, None, None].expand(mb, n_elite, steps, adim)
            elite = torch.gather(cand, 1, gi)
            mu = elite.mean(1)
            std = elite.std(1).clamp(min=0.1)
        zf = torch.cat([rollout(Zb[i:i + roll_chunk], mu[i:i + roll_chunk])
                        for i in range(0, mb, roll_chunk)])
        out[s0:s0 + mb] = _dist(zf, zg, metric).cpu().numpy()
        if verbose:
            print(f"    reach[{level}] {min(s0 + mb, M)}/{M} starts")
    return out


def render_grid(pos, rows, out, suptitle, metric,
                goals=GOALS_XY, titles=GOAL_TITLES, gridsize=24, clip_pct=70):
    """rows: list of (row_label, [cost_array_per_goal]); one column per goal."""
    nr, nc = len(rows), len(goals)
    fig, axes = plt.subplots(nr, nc, figsize=(5.0 * nc, 4.5 * nr),
                             constrained_layout=True, squeeze=False)
    for r, (label, costs) in enumerate(rows):
        for c, (gxy, gtitle) in enumerate(zip(goals, titles)):
            ax = axes[r][c]
            cost = costs[c]
            gi = goal_index(pos, gxy)
            vmin, vmax = cost.min(), np.percentile(cost, clip_pct)
            hb = ax.hexbin(pos[:, 0], pos[:, 1], C=cost, gridsize=gridsize,
                           cmap="RdYlGn_r", reduce_C_function=np.mean,
                           vmin=vmin, vmax=vmax)
            ax.axvline(WALL_X, color="k", lw=1.0, ls="--", alpha=0.5)
            ax.plot(pos[gi, 0], pos[gi, 1], marker="*", ms=20,
                    mfc="white", mec="black", mew=1.4)
            ax.set_aspect("equal")
            ax.invert_yaxis()
            if r == 0:
                ax.set_title(gtitle, fontsize=11)
            if c == 0:
                ax.set_ylabel(f"{label}\n\ny position", fontsize=10)
            ax.set_xlabel("x position", fontsize=9)
            cb = fig.colorbar(hb, ax=ax, shrink=0.85)
            cb.set_label(f"{metric.upper()} dist to goal\n(green = closer)", fontsize=8)
    fig.suptitle(suptitle, fontsize=13)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print("saved", out)
