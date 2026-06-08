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
from utils.waypoint_sampler import sample_waypoints_fixed_stride

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


if __name__ == "__main__":
    main()
