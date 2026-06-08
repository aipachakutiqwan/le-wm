# Latent analysis — runbook & cleanup log

Reproduces the latent-space analysis kept for the paper — the **macro-action linear
probe** (Fig. `macro_probe` = `macro_probe_tworoom.pdf`) — and records the
2026-06-07 trim of `qualitative analysis/latent_analysis/` down to what the paper uses.

## Prompts (the asks that drove this)

Short, paraphrased — the sequence of requests behind this analysis:

- Suggest qualitative analyses that fit a JEPA world-model + planning project.
- Estimate the lift for a latent-space t-SNE; then plan it (no code yet).
- Implement the encoder latent t-SNE — paper Fig. 9 style (physical state grid | latent, 2D-colored by position).
- Make the figure two-column / Overleaf import-ready.
- Visualize the hierarchy's *own* latent — the A_ψ macro-action space, colored by net motion.
- Add the highest-value upgrade: a linear probe (macro → net motion) with CV R²; single-column PDF, Δx/Δy stacked.
- Regenerate everything with the correct checkpoint.
- Trim the folder to what the paper uses; keep & edit the README; log it here.

## Background — full pipeline that produces this analysis

This is report qualitative analysis of the **hierarchy's learned macro-action
space**, so it needs a trained H-LeWM checkpoint. End to end:

1. **Stage 1 — flat LeWM** (ViT-tiny encoder + low-level predictor P¹ + SIGReg),
   trained from pixels on TwoRoom → Stage-1 object checkpoint:
   `python train.py data=tworoom`
2. **Stage 2 — H-LeWM**: freeze E + P¹; jointly train the action encoder A_ψ +
   high-level predictor P² (teacher-forced waypoint MSE + N(0,I) moment-matching on
   the macro-actions) → `results/hierarchical/tworooms/hierarchical_lewm_epoch_14_tworooms_object.ckpt`:
   `python train_hierarchical.py data=tworoom stage1_checkpoint=<stage1.ckpt>`
3. **This analysis** (offline — no env, no planner): load that checkpoint and run the
   two steps under *Reproduce* below.

How the kept code was arrived at: encoder/position t-SNE → macro-action (A_ψ) t-SNE
colored by motion → upgraded to the **linear probe** (macro → net motion), the
quantitative figure the paper kept. The two t-SNE explorations were dropped in the
trim (see *Deleted*).

## Reproduce (from repo root, WSL)

```bash
cd ~/le-wm
export STABLEWM_HOME=$HOME/.stable_worldmodel
CKPT=$HOME/le-wm/results/hierarchical/tworooms/hierarchical_lewm_epoch_14_tworooms_object.ckpt
DIR="qualitative analysis/latent_analysis"

# 1) extract A_ψ macro-actions -> figures/macro_action_tworoom.npz (the probe's input)
.venv/bin/python "$DIR/macro_action_tsne.py" --checkpoint "$CKPT" --device cuda
# 2) 5-fold CV linear probe -> figures/macro_probe_tworoom.pdf (+ .png preview)
.venv/bin/python "$DIR/macro_probe.py"
```

Result: `macro_probe_tworoom.pdf` — CV R²=0.89 (Δx 0.80, Δy 0.98), the paper's Fig. `macro_probe`.
Checkpoint: `results/hierarchical/tworooms/hierarchical_lewm_epoch_14_tworooms_object.ckpt` (Stage-2 H-LeWM, TwoRoom).

## Kept
- `macro_action_tsne.py` — extracts A_ψ macro-actions → `.npz` (probe input; t-SNE plot is an unused byproduct)
- `macro_probe.py` — 5-fold CV linear probe → paper figure
- `figures/macro_probe_tworoom.pdf` (paper) + `.png` (README preview)
- `README.md`

## Deleted 2026-06-07 (not used in the paper)
- `latent_grid_tsne.py` + `figures/latent_grid_tworoom.png` — encoder/position t-SNE; superseded by the `heat maps/` cost-landscape figures.
- `figures/macro_action_tworoom.png` — A_ψ macro-action t-SNE figure; paper uses the probe instead (generating script kept as the probe's data source).
- `figures/latent_grid_tworoom.npz`, `figures/macro_action_tworoom.npz` — regenerable intermediates (were never committed).
