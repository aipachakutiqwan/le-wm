# Plan: Latent cost-landscape analysis (`analysis/heat maps/`)

## Prompts (the user requests that drove this work)
1. Explain the latent cost-landscape heatmap, then generate the TwoRoom code (frozen encoder, L2); color low=green → high=red.
2. Extend it to the hierarchical model; add a flat-$P^{(1)}$ vs hierarchical-$P^{(2)}$ reachability comparison; run both.
3. Go through the plots and interpret — does the extension work, or do they show it didn't? Map each analysis to its figure.
4. Write the result as 1–2 paragraphs for the report.
5. Check the figures for completeness (axis labels/units); decide edit vs regenerate; switch the reachability grids from 3→2 columns (left room + right corner) and regenerate.
6. Using `REACHER_HANDOFF.md`: build context, then implement the Reacher cost-landscape in a new script; run it.
7. Add "latent cost vs goal offset (0–500)" plots for both envs (new scripts).
8. Build the side-by-side Reacher-vs-TwoRoom contrast; write the full results back into `REACHER_HANDOFF.md`.
9. Clean the folder to match the paper (remove dead/superseded analysis); write this plan + this prompt list.

---

> Retro/brain-dump plan. Implementing the steps below reproduces the folder's
> **final** artifacts. **Goal:** explain *why* latent planning (flat LeWM and the
> H-LeWM hierarchy) is horizon-limited, for the paper's Results section. Offline —
> frozen-encoder forward passes only (no env, no CEM). Full Reacher write-up:
> [REACHER_HANDOFF.md](REACHER_HANDOFF.md).

## Final artifacts (what should exist)
| Script (standalone) | Output(s) | Paper figure |
|---|---|---|
| `cost_landscape.py` | `cost_landscape_tworoom.png` (heatmap), `cost_landscape_tworoom_vs_distance.png` (curve) | `fig:cost_map`, `fig:cost_dist` |
| `latent_cost_vs_offset.py` | `latent_cost_vs_offset.png` (Reacher \| TwoRoom side-by-side) | `fig:cost_vs_offset` |
| `README.md` | — | — |

## Build steps
1. **TwoRoom cost landscape** — `cost_landscape.py` (standalone; `--metric l2`, `--n-frames 6000`):
   - Load a TwoRoom ckpt (`get_jepa()` → frozen encoder; only the encoder is used, so any
     TwoRoom LeWM ckpt works — default `baseline/tworoom/hierarchical_lewm_object.ckpt`).
   - Encode ~6k random frames; **L2** dist `‖E(s)−E(g)‖` to 3 goals over true `pos_agent`
     → hexbin heatmap (green=near). Sanity: linear probe `z→pos_agent` (held-out **R²≈0.99**).
   - Bin cost vs straight-line distance → curve (rises ~30 units, then plateaus).
2. **Latent cost vs eval offset** — `latent_cost_vs_offset.py` (standalone; both envs):
   - Per env — Reacher: `finger_pos`, `baseline/reacher/lewm_epoch_10_object.ckpt`;
     TwoRoom: `pos_agent`, `baseline/tworoom/lewm_epoch_9_object.ckpt` (flat stage-1).
   - Encode whole sampled episodes once; for offset Δ in `0..cap` (step 5) compute
     **within-episode** `‖E(s)−E(s+Δ)‖` (L2 latent cost) and `‖pos[s+Δ]−pos[s]‖` (true dist);
     pool, normalize per env, plot both vs Δ; mark eval offsets {25,50,75,100}.
   - **Δ capped by `ep_len`** (Reacher 201, TwoRoom ≤101) → 0–500 is impossible; doesn't matter
     (cost saturates well before the cap).

## Key decisions
- **Metric = L2** throughout = flat LeWM's planning cost (`JEPA.criterion`); L1 is only for the
  (deleted) hierarchical *reachability* figures.
- **Frozen flat stage-1 encoders** — the cost geometry is HP-independent and reused unchanged by
  H-LeWM, so it speaks to both planners.
- **Position = task metric**: Reacher fingertip `finger_pos` (±0.24); TwoRoom `pos_agent`
  ([14,208]). (`qpos` rejected — angle wrap.)
- **Eval offset** (`eval.py:117/170`): goal = state `goal_offset_steps` ahead *in the same
  episode* → offset bounded by `ep_len`. Δ distances computed within-episode via `ep_offset`/`ep_len`.

## Findings (the point)
- Latent cost = **local basin + flat plateau**; saturates beyond a short range. Position still
  decodes (R²≈0.99) → the *metric* fails, not the encoding. HP-independent.
- **TwoRoom collapses** (flat 88→12%): cost saturates *before* Δ=25 and stays flat across the whole
  sweep while true distance triples → planner scores with a dead signal.
- **Reacher stays robust** (flat 78→80%): cost also saturates, but goals stay physically tiny in an
  obstacle-free workspace → reachable anyway. Its long-offset success is **not** long-horizon capability.
- **One mechanism** drives both; the hierarchy picks subgoals by the same flat latent distance, so it
  inherits the limit → never beats flat LeWM.

## Deletions tracked (built, then removed — dead-ends / superseded)
| Removed | What it was | Why removed |
|---|---|---|
| `landscape_reachability_pair.{py,png}` | flat P¹ vs hier P² best-per-plan reach (L1) | single-plan reach can't show the *multi-step* hierarchy gain; not in paper |
| `landscape_static_vs_highlevel.{py,png}` | static cost vs P² reachable (L1) | asymmetric/low-bar; not in paper |
| `_landscape_lib.py` | shared CEM `reach_field` + `render_grid` helpers | only used by the two reachability scripts |
| `cost_landscape_reacher.{py,png}` + `…_vs_distance.png` | Reacher cost-landscape + per-Δ overlay | REACHER_HANDOFF deliverable; superseded for the paper by `latent_cost_vs_offset` (its `get_jepa`/`encode_frames` were inlined there); results kept in REACHER_HANDOFF.md |
| `cost_landscape_compare.{py,png}` | side-by-side cost-vs-distance + frac-below-knee | superseded by `latent_cost_vs_offset` |
| `__pycache__/` | bytecode | gitignored |

The five `landscape_*` / `_landscape_lib.py` were git-tracked → they appear as **deletions** in
`git status` (revert with `git checkout`); the rest were untracked. Nothing committed.

## Run (from `~/le-wm`; wrap `wsl -e bash -c "…"` if launched from Windows)
```bash
STABLEWM_HOME=$HOME/.stable_worldmodel .venv/bin/python "analysis/heat maps/cost_landscape.py" --device cuda
STABLEWM_HOME=$HOME/.stable_worldmodel .venv/bin/python "analysis/heat maps/latent_cost_vs_offset.py" --device cuda
```
Light (encoder only). `STABLEWM_HOME` **must** be set (empty in non-interactive shells → silent FileNotFound).
