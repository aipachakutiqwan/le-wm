# Temporary experiment notes — flat baseline choice (TwoRoom trajectory study)

## What we're doing here
Qualitative trajectory study on TwoRoom: plot the agent's actual (x,y) path under the
**flat** LeWM planner vs the **hierarchical** H-LeWM planner on the *same* episodes, to
*show* (not just tabulate) the planning difference. Target story: at short goal distance
both reach the goal; at long distance only the hierarchical planner routes through the
doorway to reach it. Workflow: record paths via the eval scripts (`+record_trajectories=true`)
→ plot with `viz_trajectories.py`.

## The nuance: what is the "flat" baseline?

Two valid choices of flat baseline, differing in the claim they support:

**Option 1 — flat = original paper LeWM (separate ckpt); hier = our H-LeWM.**
System-vs-system: *"does our H-LeWM beat the published LeWM?"*
- + credible external/peer-reviewed baseline; anchored to literature (~92%).
- − confounded: a win mixes *planning hierarchy* with *model/training differences*.
- − for trajectory plots, flat vs hier paths would differ for two reasons (model + planning).

**Option 2 — flat = inner JEPA extracted from our H-LeWM; hier = full H-LeWM.**
Controlled ablation: *"holding the world model fixed, what does the hierarchy add?"*
- + cleanest attribution: identical encoder + P¹ (same `self.jepa`); only single- vs two-level CEM differs.
- + ideal for trajectory plots — paths differ ONLY because of planning.
- + self-contained, reproducible from one checkpoint.
- − it's an ablation, not a "beats the baseline" claim.

## Why they mostly coincide for us
epoch-14 H-LeWM was trained on the **frozen paper Stage-1**, so the inner JEPA we extract
for Option 2 ≈ the paper LeWM. Sizes corroborate: flat LeWM ≈ 72 MB, H-LeWM ≈ 117 MB
(the ~45 MB gap = Stage-2 macro components A_ψ + P²).

## DECISION
**Run experiments for BOTH cases** (Option 1 and Option 2) so we can report the clean
ablation *and* the published-baseline comparison.

## TODO / verification
- [ ] Compare epoch-14 inner-JEPA `state_dict` vs `~/.stable_worldmodel/tworoom/lewm/lewm_object.ckpt`.
      identical  -> Option 1 ≈ Option 2 (one flat run serves both claims).
      different  -> Stage-2 touched P¹; report Option 1 separately.

## Checkpoints on disk
- H-LeWM (hier + Option-2 flat): `results/hierarchical/tworooms/hierarchical_lewm_epoch_14_tworooms_object.ckpt` (117 MB)
- paper flat LeWM (Option 1):    `~/.stable_worldmodel/tworoom/lewm/lewm_object.ckpt` (72 MB)
- other flat ckpt:               `baseline/tworoom/lewm_epoch_9_object.ckpt` (72 MB) — provenance TBD

## Run config
- TwoRoom, d = goal_offset ∈ {25, (50), 75}, eval_budget = 2×d, seed=42.
- This chat = trajectory plots only. First pass: **Option 2, d=25, N=10**.
- Outputs: `analysis/path_trajectories/runs/{flat,hier}_dXX/`; figures in `analysis/path_trajectories/runs/figures/`.

## Runs log

### Run 1 — Option 2, d=25, N=10, seed=42, eval_budget=50  (2026-06-02)
Checkpoint (both planners): `results/hierarchical/tworooms/hierarchical_lewm_epoch_14_tworooms_object.ckpt`.
- **flat** (inner JEPA, single-level CEM): **10/10 = 100%**
- **hier** (full H-LeWM, two-level CEM; H_high=1, h_low=3, outer_std=2.5, inner_iters=30): **7/10 = 70%**, ~80 s.
- hier failures = the **3 farthest** goals (init_dist 53/63/70; mean init_dist: success 30.0 vs fail 61.9).
- Figure: `runs/figures/trajectories_d25.png` — flat ✓ on all 10; hier ✗ on ep 953 / 6989 / 8601.
- Read: at short range flat ≥ hier (expected); distance-to-goal already predicts hier failure. No
  "only-hier-reaches" contrast possible at d=25 (flat never fails) → that contrast needs **d=75**.
