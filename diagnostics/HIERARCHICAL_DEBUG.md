# Hierarchical LeWM — planning failure investigation

## CONCLUSION (2026-05-27)

The 20% has **two fixed execution bugs plus one dominant model-quality cause**.

**Decisive evidence:** a random policy scored **50%** on the same 10 episodes vs the
planner's **20%** — the planner is *worse than random* and only wins the two episodes
that start essentially on the goal (init_dist 12.9, 13.9). Successes correlate purely
with starting distance (mean init_dist success 13.4 vs fail 46.1), i.e. the planner
barely controls the agent.

Bugs found and fixed (necessary, but did **not** move eval on their own):
1. **Action scrambling across envs** in `HierarchicalPolicy.get_action`: reshape
   `(n_envs*fs, base) -> (fs, n_envs, base)` mixed primitives between envs when
   `num_envs > 1`. Fixed: `reshape(n_envs, fs, base)` → inverse_transform → `transpose(1,0,2)`.
2. **Inner CEM std 0.1** vs real action std ~0.86 → low-level under-exploration. Fixed:
   exposed `plan.inner_std`, default 0.5.

**Dominant cause — `P²` is a weak goal-directed dynamics model.** Optimizing macro-actions
directly against the true goal latent, the best the high-level rollout closes is only
**7–45% (typically ~15%) of the goal distance in 3 steps**. Consequences measured on real
frames:
- planner commands actions of magnitude ~0.12–0.30 vs real actions ~0.8 (**3–7× too small**);
- the agent crawls toward subgoals that don't lead to the goal → only near-goal starts win.

Both P² and P¹ *can* move the agent when fed wide/full-scale inputs (P² reaches 234 vs
real-step 211; P¹ reaches 269 vs real 224) — so the capacity exists; the **learned
goal-directed multi-step dynamics do not**. Root: one-step teacher forcing (no penalty on
compounding error; H=3 rollout explains ~15–28%) + overfitting (train TF 0.17 vs held-out
0.84) + the 4-dim macro-action bottleneck.

**Verdict: planning-side tuning cannot fix this — retraining is required** (multi-step
rollout loss, wider `latent_action_dim`, overfitting control). The two bug fixes should be
kept regardless (multi-env eval correctness).

---

## Context

Stage-2 `HierarchicalLeWM` (run `20260526_170216`, TwoRoom, 50 epochs) plans at only
**~20% success** with `plan_hierarchical.py`, far below flat LeWM. Crucially, the 20%
was **invariant** to every planning knob tried:

| change | success |
|---|---|
| baseline (`H_high=3, h_low=10`) | 20% |
| `h_low=5` | 20% |
| `outer_samples=1024, inner_samples=512, outer_iters=8, inner_iters=8` | 20% |
| `H_high=1` | 20% |
| `H_high=2` | 20% |

Training itself looked healthy: teacher-forced loss fell 1.75 → 0.17 over 50 epochs,
variance penalty saturated to 0 by epoch ~6.

## Why we're checking this

Low teacher-forced loss + failed planning is a classic world-model smell. When the
success rate is *identical* across substantially different planning configs, the
planner's actions probably aren't controlling the agent — i.e. 20% is likely "free"
success (episodes starting near the goal). We isolate the cause by measuring the
model's own behaviour on real, on-manifold latents, decoupled from the env loop.

Hypotheses:

- **H1 — action-conditioning collapse.** `P²` predicts the next waypoint from the
  previous waypoint latent and ignores/under-uses the 4-dim macro-action. The planner
  steers *only* through macro-actions, so this removes all control authority.
- **H2 — outer-prior mismatch.** The outer CEM samples macro-actions from `N(0,1)`,
  but `A_ψ` may emit a very different distribution; the planner would explore a region
  `P²` never saw.
- **AR — high-level rollout fidelity.** TF loss is one-step; planning uses
  autoregressive rollout. Compounding error invisible to the TF curve.
- **LOW — low-level path.** Does `P¹` (`_rollout_low`) reach subgoals, and is the inner
  CEM `std=0.1` matched to the real action scale?

## How to run

```bash
STABLEWM_HOME=/path/to/cache \
python diagnostics/hierarchical_probe.py \
    --checkpoint /stablewm-home/.../hierarchical_lewm_object.ckpt \
    --dataset tworoom --device cuda --goal-offset 25
```

The probe encodes real dataset frames with the model's frozen JEPA, then runs all
checks. No env loop / no W&B needed. Set `--goal-offset` to match `eval.goal_offset_steps`
(25 for TwoRoom). Runs in well under a minute on a GPU.

### Which output reveals the dominant cause (weak `P²`)

Read these two blocks; together they are the verdict:

1. **`[GOAL] ... ** DECISIVE **`** — replicates the planner on real (start, goal+offset)
   pairs. The signature of a weak high-level model is:
   - `MEAN goal closure` low (≈15–30%): even optimizing macro-actions directly against the
     true goal latent, the best rollout barely approaches it.
   - `MEAN planner |action|` ≪ `real |action|` (we see ~0.13 vs ~1.15, ~9×): the planner
     commands sluggish actions, so the agent crawls and only near-goal starts succeed.

   Example output:
   ```
   [GOAL] end-to-end planner on real (start, goal+25) pairs  ** DECISIVE **
     MEAN goal closure = 30.2%   MEAN planner |action| = 0.130  (real |action| ~ 1.153)
     -> low closure + small action vs real scale => P^(2) is a weak goal-directed model
   ```

2. **`[AR]`** — `P²` autoregressive fidelity by horizon. Confirms *why* closure is low:
   accuracy collapses past ~2 steps (H=1 ≈ 74%, H=3 ≈ 15–28%), so multi-step plans drift.

The other blocks rule out alternatives: `[H2]`/`[H2b]` show the CEM prior is not the
bottleneck (iterative CEM compensates), and `[LOW]` shows the low-level path and inner-CEM
std are secondary. The combination — low goal closure + tiny actions + AR collapse, with
priors/low-level ruled out — is what pins the cause on `P²`'s learned dynamics.

## Observations (run `20260526_170216`, 24–48 windows)

**H2 — `A_ψ` macro-action distribution vs the `N(0,1)` planning prior**
```
per-dim mean = [ 6.97, -2.02, -2.80,  2.71]   (|mean| avg 3.6)
per-dim std  = [ 0.89,  2.06,  1.23,  1.86]
```
The real macro-actions are far off-center (dim-0 mean ≈ 7, ~7σ from where `N(0,1)`
samples). The prior is mis-specified.

**H2b — but iterative CEM compensates** (this is the key nuance):
```
CEM prior N(0,1)                    -> goal-dist reduction 39.6%
CEM prior N(mu_real, 2*std_real)    -> goal-dist reduction 36.6%
```
Iterative CEM migrates from `N(0,1)` toward the real region on its own. A data-driven
prior does **not** help. → **H2 is a red herring; the outer prior is not the bottleneck.**
(Consistent with success being flat under more samples/iters.)

**AR — `P²` autoregressive fidelity with the TRUE macro-actions** (variance explained):
```
H=1  74%      <- usable
H=2  45%
H=3  28%      <- planner default ran here
H=4  14%      <- noise
```
`P²` is an OK 1-step predictor but degrades fast under autoregressive rollout. The root
of this is one-step teacher forcing, which never penalises compounding error.

**Held-out vs train 1-step loss:** held-out 1-step TF MSE ≈ **0.84** vs train **0.17**
→ `P²` is **overfitting** (it's a 6-layer/16-head/2048-MLP transformer).

**LOW — low-level path**
```
P¹ rollout fidelity (true actions, 5 steps): ~70% explained   (decent)
real effective-action per-dim std ≈ 0.86, but inner CEM std = 0.1  (~8x too tight)
inner CEM reach real subgoal vs prior std:
    std=0.1 -> 35%
    std=0.3 -> 59%
    std=0.5 -> 69%   <- best
    std=1.0 -> 56%
```
The inner CEM under-explores the action space by ~8×. Widening `std` 0.1 → 0.5 lifts
subgoal-reaching 35% → 69% (capped by P¹'s ~70% rollout fidelity).

## Interpretation

Two independent, confirmed defects — neither fixable by tuning the existing planning args:

1. **`P²` autoregressive fidelity collapses past ~2 macro-steps** (root: pure one-step
   teacher forcing + overfitting). Even `H_high=1` didn't recover success, which means
   this is necessary but **not sufficient** to explain 20%.
2. **Inner CEM prior `std=0.1` is ~8× too small** vs real action std 0.86 → the low-level
   planner can barely move the agent.

The flat-20%-everywhere signature additionally suggests the executed actions are nearly
ineffective end-to-end, so **a pipeline/execution bug is not yet ruled out** (e.g. the
`HierarchicalPolicy` frameskip queue, action denormalisation, or goal encoding). This
must be checked before/along with retraining, because no model improvement will help if
the planned actions don't reach the env correctly.

## Next steps (ranked)

**Confirm it's not a pipeline bug (do first — cheap, decisive):**
1. Instrument `plan_hierarchical.py` eval to log per-episode initial distance-to-goal.
   If the ~2 successful episodes start near the goal, the planner contributes ≈0 → look
   for an execution bug in `HierarchicalPolicy.get_action` (frameskip action queue,
   `scaler.inverse_transform` denormalisation, init vs goal encoding).

**No-retrain planner fixes:**
2. **Widen inner CEM std 0.1 → 0.5** in `hierarchical_plan.py` (or expose as
   `plan.inner_std`). Confirmed to ~2× low-level subgoal reach. Re-run planning.
3. Keep `H_high` small (1–2) with receding-horizon replanning to stay in `P²`'s
   accurate regime.

**Retrain fixes (the real solution), ranked:**
4. **Multi-step autoregressive rollout loss / scheduled sampling** for `P²` — unroll k
   steps feeding predictions back and supervise against future waypoints. Directly targets
   the AR collapse. Expose depth as `stage2.rollout_steps`.
5. **`latent_action_dim` 4 → 8/16** — widen the macro-action bottleneck.
6. **Curb `P²` overfitting** — weight decay / smaller `high_depth`; select checkpoints by
   rollout/planning metric, not 1-step TF loss (val logging + per-epoch checkpoints are
   now in place to support this).
7. Optionally **normalise the macro-action space** so `A_ψ` outputs ≈ `N(0,1)`, making the
   outer prior well-specified (cosmetic given H2b, but cleaner).
