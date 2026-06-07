# path_trajectories — flat vs H-LeWM trajectory study (prompts · reproduce · cleanup log)

## Prompts (the asks)
- Plot flat LeWM vs H-LeWM rollout trajectories on TwoRoom at offset `D` — same checkpoint, same episodes; record then overlay.
- Run flat (fast) then H-LeWM (slow → **run in background**) at `D=25, N=10`; then plot the comparison.
- Sweep `D∈{25,50,75}`, budget `=2×D`, flat vs hier; cherry-pick a 2×2 (both reach @ short, only hier @ long).
- Explain how `eval_budget` is consumed for flat vs hierarchical planning.
- Use the **Option-2** baseline: extract the inner JEPA from our H-LeWM so both planners share the encoder (clean ablation).
- Given the paper, remove repo artifacts it doesn't reference (videos, npz, logs).

## Reproduce the analysis (from `~/le-wm`, serial — hier replans every step, so it's slow)
Same checkpoint for both planners (Option 2): `models/hierarchical_lewm_epoch_14_tworooms_object.ckpt`
```bash
CKPT=$HOME/le-wm/models/hierarchical_lewm_epoch_14_tworooms_object.ckpt
RUNS="$HOME/le-wm/qualitative analysis/path_trajectories/runs"
ln -sfn "$RUNS" "$HOME/.lewm_runs"               # no-space alias for Hydra args (path has a space)

D=25; B=$((2*D)); N=10                            # repeat with D=50 (B=100), D=75 (B=150)
mkdir -p "$RUNS/flat_d$D" "$RUNS/hier_d$D" "$RUNS/figures"
ln -sf "$CKPT" "$RUNS/flat_d$D/ckpt_object.ckpt"
ln -sf "$CKPT" "$RUNS/hier_d$D/ckpt.ckpt"

# flat: single-level CEM (AutoCostModel extracts inner JEPA)
STABLEWM_HOME=$HOME/.stable_worldmodel .venv/bin/python eval.py --config-name=tworoom.yaml \
  policy=$HOME/.lewm_runs/flat_d$D/ckpt +cache_dir=$HOME/.stable_worldmodel \
  eval.num_eval=$N eval.goal_offset_steps=$D eval.eval_budget=$B seed=42 \
  +record_trajectories=true +traj_npz=trajectories_flat.npz

# hier: two-level CEM (TUNED HPs — NOT the stale config defaults)
STABLEWM_HOME=$HOME/.stable_worldmodel .venv/bin/python plan_hierarchical.py \
  checkpoint=$HOME/.lewm_runs/hier_d$D/ckpt.ckpt device=cuda seed=42 \
  eval.num_eval=$N eval.goal_offset_steps=$D eval.eval_budget=$B \
  plan.H_high=1 plan.h_low=3 plan.outer_std=2.5 plan.inner_std=1.0 \
  plan.outer_samples=512 plan.inner_samples=256 plan.outer_iters=5 plan.inner_iters=30 \
  +record_trajectories=true +traj_npz=trajectories_hier.npz

# plot flat vs hier paths
.venv/bin/python "qualitative analysis/path_trajectories/viz_trajectories.py" \
  --flat "$RUNS/flat_d$D/trajectories_flat.npz" --hier "$RUNS/hier_d$D/trajectories_hier.npz" \
  --out "$RUNS/figures/trajectories_d$D.png"
```

### Paper diagnostic figures (frozen encoder, no env — fast)
- `heat maps/cost_landscape.py --device cuda` → `cost_landscape_tworoom{,_vs_distance}.png`
- `heat maps/latent_cost_vs_offset.py --device cuda` → `latent_cost_vs_offset.png`
- `latent_analysis/macro_probe.py` → `macro_probe_tworoom.pdf` (see its README for flags)

## Deletions — cleanup 2026-06-07 (all regenerable / paper-unreferenced; 0 git-tracked)
- `wandb/` (25M, also on W&B server), `outputs/` (hydra logs), all `__pycache__/`  → ~26M
- all 590 `*.mp4` rollout videos (`long_horizon_experiments/`, `path_trajectories/runs/`)  → ~15M
- all 8 `*.npz` trajectory recordings (`path_trajectories/runs/{flat,hier}_d{25,50,75,100}`)
- **Kept:** code, configs, `models/…epoch_14` ckpt, paper figures + `result.txt`, `.claude/`, `temp_exp.md`
- **Reclaimed ≈ 41M.** Still removable: `baseline/tworoom/hierarchical_lewm_object.ckpt` (117M, superseded by epoch_14), `baseline/{cube,pusht}` flat ckpts (144M, re-downloadable).
