"""Diagnostics for a trained HierarchicalLeWM that plans poorly.

Motivation
----------
Stage-2 hierarchical planning scored ~20% on TwoRoom while flat LeWM scores much
higher, and the 20% was *invariant* to every planning knob (H_high, h_low, CEM
samples/iters). This script isolates where the failure lives by measuring the
model's own behaviour directly, decoupled from the env loop:

  H1  action-conditioning collapse : does P^(2) actually use the macro-action,
                                      or predict the next waypoint from z alone?
  H2  outer-prior mismatch          : does A_psi's macro-action distribution match
                                      the planner's N(0,1) sampling prior?
  AR  high-level rollout fidelity   : does P^(2) stay accurate over multiple
                                      autoregressive steps (planning runs at H>=1)?
  LOW low-level path                : does P^(1) (_rollout_low) reach subgoals, and
                                      is the inner CEM std matched to action scale?

All probes operate on real, on-manifold latents obtained by encoding dataset
frames with the model's frozen JEPA encoder.

Usage
-----
python "analysis/diagnostics/hierarchical_probe.py" \
    --checkpoint /stablewm-home/.../hierarchical_lewm_object.ckpt \
    --dataset tworoom --device cuda

Requires STABLEWM_HOME to point at the dataset cache (same as training/eval).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# repo root (diagnostics -> analysis -> le-wm) holds hierarchical_plan, etc.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import stable_worldmodel as swm
from sklearn import preprocessing
from hierarchical_plan import cem, plan
from waypoint_sampler import sample_waypoints_fixed_stride

# ImageNet normalisation — must match img_transform() in plan_hierarchical.py
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def load_windows(dataset, num_windows, num_steps, frameskip, wp_idx, device,
                 seed=0, action_scaler=None):
    """Sample `num_windows` trajectory windows; return waypoint latents + actions.

    If `action_scaler` is given (sklearn StandardScaler fit on the dataset's raw
    action column), the returned actions are z-scored to match the distribution
    A_ψ / P¹ were trained on. Pass None to get raw actions (off-distribution —
    the historical bug).

    Returns
    -------
    wp_emb : (M, W, D)  encoded waypoint latents (W = len(wp_idx))
    act    : (M, T, A)  effective (frameskip-folded) actions over the window
    """
    idxs = np.random.default_rng(seed).choice(len(dataset), size=num_windows, replace=False)
    pix, act = [], []
    for i in idxs:
        s = dataset[int(i)]
        pix.append(s["pixels"][wp_idx])   # (W, 3, H, W) uint8
        act.append(s["action"])           # (T, A)
    pix = torch.stack(pix).float() / 255.0
    pix = ((pix - _MEAN) / _STD).to(device)
    act = torch.stack(act)                # (M, T, A_eff)
    if action_scaler is not None:
        # A_eff = frameskip × base_dim; scaler was fit on base_dim raw actions.
        M, T, A_eff = act.shape
        base = action_scaler.n_features_in_
        arr = act.cpu().numpy().reshape(M, T, A_eff // base, base)
        arr = (arr - action_scaler.mean_) / action_scaler.scale_
        act = torch.from_numpy(arr.reshape(M, T, A_eff)).float()
    return pix, act.to(device)


def encode(model, pix):
    with torch.no_grad():
        return model.jepa.encode({"pixels": pix})["emb"]   # (M, W, D)


def real_macro_actions(model, act, wp_idx):
    """Encode each inter-waypoint action chunk with A_psi -> (M, n_seg, d_L)."""
    segs = []
    for k in range(len(wp_idx) - 1):
        chunk = torch.nan_to_num(act[:, wp_idx[k]:wp_idx[k + 1]], 0.0)
        with torch.no_grad():
            segs.append(model.action_encoder_high(chunk))
    return torch.stack(segs, 1)


# ──────────────────────────────────────────────────────────────────────────────
# H2 — outer prior mismatch
# ──────────────────────────────────────────────────────────────────────────────
def probe_macro_distribution(model, macro_real):
    dL = model.latent_action_dim
    flat = macro_real.reshape(-1, dL)
    print("\n[H2] A_psi macro-action distribution vs planning prior N(0,1)")
    print(f"  per-dim mean : {flat.mean(0).cpu().numpy().round(3)}")
    print(f"  per-dim std  : {flat.std(0).cpu().numpy().round(3)}")
    print(f"  |mean| avg = {flat.mean(0).abs().mean():.3f}   std avg = {flat.std(0).mean():.3f}")
    print("  -> if |mean| >> 1 or std != 1, the CEM N(0,1) prior is off-distribution")


# ──────────────────────────────────────────────────────────────────────────────
# H1 — control authority (does varying the macro-action move the prediction?)
# ──────────────────────────────────────────────────────────────────────────────
def probe_control_authority(model, wp_emb, macro_real, H_high=3, K=512):
    dL = model.latent_action_dim
    D = wp_emb.shape[-1]
    sigma_z = wp_emb.reshape(-1, D).std(0).mean().item()
    z_init = wp_emb[:, 0]
    z_goal = wp_emb[:, H_high]
    d0 = (z_init - z_goal).abs().sum(-1)
    rmean = macro_real.reshape(-1, dL).mean(0)
    rstd = macro_real.reshape(-1, dL).std(0)
    M = wp_emb.shape[0]

    print(f"\n[H1] control authority (one-shot sampling, K={K}, sigma_z={sigma_z:.3f})")
    for label, mu, sd in [("N(0,1)", 0.0, 1.0), ("real macro scale", rmean, rstd)]:
        reach, red = [], []
        for m in range(M):
            cand = torch.randn(K, H_high, dL, device=wp_emb.device) * (sd if torch.is_tensor(sd) else sd) + (mu if torch.is_tensor(mu) else mu)
            zi = z_init[m:m + 1].expand(K, -1)
            with torch.no_grad():
                zl = model._rollout_high(zi, cand)[:, -1]
            reach.append(zl.std(0).mean().item())
            dist = (zl - z_goal[m:m + 1]).abs().sum(-1)
            red.append(((d0[m] - dist.min()) / d0[m]).item())
        print(f"  prior={label:18s} reachable spread/sigma_z={np.mean(reach)/sigma_z:.3f}  "
              f"best goal-dist reduction={100*np.mean(red):.1f}%")


def probe_outer_cem(model, wp_emb, macro_real, H_high=3, iters=8, samples=1024):
    """Run the *actual* iterative CEM and measure achievable goal-distance reduction."""
    dL = model.latent_action_dim
    z_init, z_goal = wp_emb[:, 0], wp_emb[:, H_high]
    d0 = (z_init - z_goal).abs().sum(-1)
    rmean = macro_real.reshape(-1, dL).mean(0)
    rstd = macro_real.reshape(-1, dL).std(0)
    M = wp_emb.shape[0]
    dev = wp_emb.device

    def run(mu0, sd0, label):
        red = []
        for m in range(M):
            zi, zg = z_init[m], z_goal[m]

            def cost(c):
                with torch.no_grad():
                    zl = model._rollout_high(zi.unsqueeze(0).expand(c.shape[0], -1), c)[:, -1]
                return (zl - zg).abs().sum(-1)

            best = cem(cost, mu0.clone(), sd0.clone(), n_samples=samples, n_iters=iters)
            with torch.no_grad():
                zl = model._rollout_high(zi.unsqueeze(0), best.unsqueeze(0))[:, -1]
            red.append(((d0[m] - (zl - zg).abs().sum()) / d0[m]).item())
        print(f"  CEM prior={label:28s} goal-dist reduction={100*np.mean(red):.1f}%")

    print(f"\n[H2b] iterative CEM ({iters} iters, {samples} samples) — does prior matter?")
    run(torch.zeros(H_high, dL, device=dev), torch.ones(H_high, dL, device=dev), "N(0,1)")
    run(rmean.unsqueeze(0).expand(H_high, -1), (2 * rstd).unsqueeze(0).expand(H_high, -1),
        "N(mu_real, 2*std_real)")


# ──────────────────────────────────────────────────────────────────────────────
# AR — high-level rollout fidelity per horizon
# ──────────────────────────────────────────────────────────────────────────────
def probe_high_ar_fidelity(model, wp_emb, macro_real):
    z_init = wp_emb[:, 0]
    print("\n[AR] P^(2) autoregressive fidelity with TRUE macro-actions")
    print("     (explained = 1 - MSE(pred,true)/MSE(z_init,true); planning uses H_high)")
    for H in range(1, wp_emb.shape[1]):
        with torch.no_grad():
            pred = model._rollout_high(z_init, macro_real[:, :H])[:, -1]
        tgt = wp_emb[:, H]
        err = (pred - tgt).pow(2).mean().item()
        copy = (z_init - tgt).pow(2).mean().item()
        print(f"  H={H}  MSE={err:.3f}  baseline(stay)={copy:.3f}  explained={100*(1-err/copy):5.1f}%")


def probe_tf_holdout(model, wp_emb, macro_real):
    """All-positions one-step teacher-forced MSE on held-out windows.

    Mirrors what `train_hierarchical_lewm` logs as `stage2/val_loss`: predicts
    waypoints 1..n_seg from the TRUE previous waypoints (single parallel causal
    pass, no AR feedback) and averages MSE across all positions. Use this to
    compare a checkpoint against `stage2/val_loss`, or to get the metric for
    checkpoints trained without val tracking (see METRICS.md).
    """
    with torch.no_grad():
        pred = model.high_predictor(wp_emb[:, :-1], macro_real)
    tf_mse = (pred - wp_emb[:, 1:]).pow(2).mean().item()
    print(f"\n[TF] one-step teacher-forced MSE (all positions, held-out): {tf_mse:.4f}")
    print(f"     matches `stage2/val_loss` for trained models; comparable across runs")


# ──────────────────────────────────────────────────────────────────────────────
# LOW — low-level path: P^(1) fidelity + inner CEM std
# ──────────────────────────────────────────────────────────────────────────────
def probe_low_level(model, wp_emb, act, wp_idx):
    A = model.action_dim
    dev = wp_emb.device
    z_init, subgoal = wp_emb[:, 0], wp_emb[:, 1]
    seg_len = wp_idx[1] - wp_idx[0]
    real_acts = torch.nan_to_num(act[:, wp_idx[0]:wp_idx[1]], 0.0)

    with torch.no_grad():
        zr = model._rollout_low(z_init, real_acts)
    err = (zr - subgoal).pow(2).mean().item()
    copy = (z_init - subgoal).pow(2).mean().item()
    print(f"\n[LOW] P^(1) rollout fidelity (true actions, {seg_len} steps): "
          f"explained={100*(1-err/copy):.1f}%")

    flat = torch.nan_to_num(act.reshape(-1, A), float("nan"))
    real_std = np.nanstd(act.reshape(-1, A).cpu().numpy(), axis=0)
    print(f"  real effective-action per-dim std avg = {np.nanmean(real_std):.3f}  "
          f"(inner CEM default std=0.1 -> under-explores if these differ)")

    M = wp_emb.shape[0]

    def inner(std0):
        red = []
        for m in range(M):
            z0, g = wp_emb[m, 0], wp_emb[m, 1]
            d0 = (z0 - g).abs().sum()

            def cost(c):
                with torch.no_grad():
                    zf = model._rollout_low(z0.unsqueeze(0).expand(c.shape[0], -1), c)
                return (zf - g).abs().sum(-1)

            best = cem(cost, torch.zeros(seg_len, A, device=dev),
                       torch.full((seg_len, A), std0, device=dev), 512, 8)
            with torch.no_grad():
                zf = model._rollout_low(z0.unsqueeze(0), best.unsqueeze(0))
            red.append(((d0 - (zf - g).abs().sum()) / d0).item())
        return 100 * np.mean(red)

    print("  inner CEM reach real subgoal vs prior std:")
    for s in [0.1, 0.3, 0.5, 1.0]:
        print(f"    std={s:<4} -> distance reduction={inner(s):.1f}%")


# ──────────────────────────────────────────────────────────────────────────────
# GOAL — end-to-end planner behaviour on real (start, goal) pairs  ** DECISIVE **
# ──────────────────────────────────────────────────────────────────────────────
def probe_goal_reachability(model, dataset_name, device, goal_offset=25, n_pairs=8,
                            H_high=3, h_low=10, inner_std=0.5, seed=0):
    """Replicate the planner on real (start, goal=offset-ahead) pairs.

    This is the decisive probe for the dominant failure. It reports, per pair:
      - goal closure: how much of |z_init - z_goal| the best macro-action sequence
        can close (weak P^(2) => only ~15%).
      - action magnitude: the planner's executed (denormalised) first action vs the
        real action scale (a sluggish planner => actions several-fold too small).
    """
    flat = swm.data.HDF5Dataset(dataset_name, keys_to_cache=["action", "proprio"])
    acol = np.asarray(flat.get_col_data("action"))
    acol = acol[~np.isnan(acol).any(axis=1)]
    scaler = preprocessing.StandardScaler().fit(acol)
    real_abs = np.abs(acol).mean(0)

    mean = _MEAN.view(1, 3, 1, 1)
    std = _STD.view(1, 3, 1, 1)

    def enc(row):
        px = ((flat[row]["pixels"].float() / 255.0 - mean) / std).unsqueeze(0).to(device)
        with torch.no_grad():
            return model.jepa.encode({"pixels": px})["emb"][:, -1].squeeze(0)

    dL = model.latent_action_dim
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(flat) - goal_offset - 1, size=n_pairs, replace=False)

    print(f"\n[GOAL] end-to-end planner on real (start, goal+{goal_offset}) pairs  ** DECISIVE **")
    print(f"  real action |mean abs| per-dim = {real_abs.round(3)}  (planner actions should be comparable)")
    closures, act_norms = [], []
    for r in rows:
        zi, zg = enc(int(r)), enc(int(r) + goal_offset)
        d0 = (zi - zg).abs().sum()

        def cost(c):
            with torch.no_grad():
                zl = model._rollout_high(zi.unsqueeze(0).expand(c.shape[0], -1), c)[:, -1]
            return (zl - zg).abs().sum(-1)

        best = cem(cost, torch.zeros(H_high, dL, device=device),
                   torch.ones(H_high, dL, device=device), 512, 5)
        with torch.no_grad():
            zf = model._rollout_high(zi.unsqueeze(0), best.unsqueeze(0))[:, -1].squeeze(0)
        closure = float((1 - (zf - zg).abs().sum() / d0) * 100)

        eff = plan(model, zi, zg, H_high=H_high, h_low=h_low, inner_std=inner_std).cpu().numpy()
        prim0 = scaler.inverse_transform(eff.reshape(-1, scaler.n_features_in_))[0]
        act_norm = float(np.linalg.norm(prim0))
        closures.append(closure)
        act_norms.append(act_norm)
        print(f"  row={int(r):7d} |zi-zg|={float(d0):7.1f}  best-macro goal closure={closure:5.1f}%  "
              f"planner |action|={act_norm:.3f}")
    print(f"  MEAN goal closure = {np.mean(closures):.1f}%   MEAN planner |action| = {np.mean(act_norms):.3f}  "
          f"(real |action| ~ {np.linalg.norm(real_abs):.3f})")
    print("  -> low closure + small action vs real scale => P^(2) is a weak goal-directed model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", default="tworoom")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-windows", type=int, default=32)
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--n-waypoints", type=int, default=4)
    ap.add_argument("--goal-offset", type=int, default=25,
                    help="goal distance (steps) for the decisive end-to-end probe; match eval")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault("STABLEWM_HOME", os.environ.get("STABLEWM_HOME", "/stablewm-home"))
    torch.manual_seed(args.seed)

    model = torch.load(args.checkpoint, map_location=args.device, weights_only=False).eval()
    dataset = swm.data.HDF5Dataset(
        name=args.dataset, num_steps=args.num_steps, frameskip=args.frameskip,
        keys_to_load=["pixels", "action", "proprio"],
        keys_to_cache=["action", "proprio"], transform=None,
    )
    wp_idx = sample_waypoints_fixed_stride(args.num_steps, N=args.n_waypoints).tolist()

    print(f"checkpoint={args.checkpoint}")
    print(f"device={args.device}  dataset={args.dataset}  windows={args.num_windows}")
    print(f"waypoint indices={wp_idx}  latent_action_dim={model.latent_action_dim}  "
          f"action_dim={model.action_dim}  history_size={model.history_size}")

    # Fit the same action scaler the training/eval pipeline uses (z-scores actions
    # before they reach A_ψ / P¹). Feeding raw actions to those modules yields
    # off-distribution behaviour and inflated MSE — see METRICS.md.
    flat_ds = swm.data.HDF5Dataset(args.dataset, keys_to_cache=["action", "proprio"])
    acol = np.asarray(flat_ds.get_col_data("action"))
    acol = acol[~np.isnan(acol).any(axis=1)]
    action_scaler = preprocessing.StandardScaler().fit(acol)
    print(f"action scaler  mean={action_scaler.mean_.round(3)}  scale={action_scaler.scale_.round(3)}")

    pix, act = load_windows(dataset, args.num_windows, args.num_steps, args.frameskip,
                            wp_idx, args.device, seed=args.seed,
                            action_scaler=action_scaler)
    wp_emb = encode(model, pix)
    macro_real = real_macro_actions(model, act, wp_idx)

    probe_macro_distribution(model, macro_real)
    probe_control_authority(model, wp_emb, macro_real, H_high=args.n_waypoints)
    probe_outer_cem(model, wp_emb, macro_real, H_high=args.n_waypoints)
    probe_high_ar_fidelity(model, wp_emb, macro_real)
    probe_tf_holdout(model, wp_emb, macro_real)
    probe_low_level(model, wp_emb, act, wp_idx)
    probe_goal_reachability(model, args.dataset, args.device,
                            goal_offset=args.goal_offset, seed=args.seed)


if __name__ == "__main__":
    main()
