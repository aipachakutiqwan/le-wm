# Issues log

---

# Issue 1: our waypoint sampling has high within-sample span variance vs the paper — **RESOLVED 2026-05-27**

**Resolution:** Stage-2 retraining (`bnil0skdx`) used `sample_waypoints_fixed_stride`, matching PLDM Appendix B.4 exactly. Combined with the other training-side fixes the model retrained cleanly (`L_tf=0.149`, monotonic descent over 10 epochs).

**Honest follow-up:** the sampler swap was bundled with other training-side knobs (lr, KL term, span, d_L, n_waypoints, cosine schedule). The apples-to-apples eval (`bfykxoeiq`) attributes **+10 pts** of success rate to *all* the training-side changes combined on N=10 (noisy). The sampler swap's individual contribution is not isolated and may be smaller than `issues.md` originally implied — the dominant remaining bottleneck turned out to be the inner-planner CEM iteration count (see Issue 2 below), not the macro-action representation.

---



## TL;DR

Our Stage-2 waypoint sampler picks N=3 interior indices **uniformly at random** with no minimum gap. That gives much higher within-sample variance in segment length than the HWM paper actually uses, especially compared to PLDM (the closest analog to LeWM).

## What the paper does

| HWM backbone | Waypoint spec | Within-sample span variance |
| --- | --- | --- |
| **PLDM** (closest analog to LeWM — small JEPA from pixels) | **fixed stride 10 raw steps** | zero |
| DINO-WM (Push-T) | segments in [25, 70] steps | ≤ 2.8× |
| VJEPA2-AC (Franka) | segments in [0.33, 4] s | controlled range |

PLDM uses **deterministic uniform spacing**. The other two use controlled ranges, not pure random.

## What our code does

`sample_waypoints(T=20, N=3)` draws 3 interior indices via `torch.randperm(T-2)[:N]`. No `min_gap`, no stride control. On a 20-frame sample:

- typical draw `[0, 4, 11, 16, 19]` → segments of `[4, 7, 5, 3]` frames (ratio 2.3×)
- pathological-but-legal draw `[0, 1, 2, 3, 19]` → segments of `[1, 1, 1, 16]` frames (ratio **16×**)

So a single training sample can ask `A_psi` to summarize anywhere from 1 to ~16 effective actions, and ask `P²` to bridge anywhere from ~5 to ~80 raw env steps — **all in the same forward pass**.

## Why it matters

- The same 4-dim macro-action latent has to represent both "1-step nudge" and "16-step traversal across the room". The transformer in `A_psi` handles variable L architecturally, but the latent code ends up ambiguous about *how far* a macro-action actually moves you.
- At test time the outer CEM samples macro-actions from a fixed `N(0, I)`. It has no notion of "this macro represents 5 steps vs 15 steps". If training never settled on a consistent span, the planner's macro-action distribution is hard to calibrate.
- The original design (plans/plan.md, Step 4) called for `mode ∈ {random, fixed_stride, endpoints}` plus a `min_gap` parameter precisely to bound this — only `random` got implemented, with no `min_gap`.

## Two ways to fix

1. **Match PLDM (recommended)** — fixed stride. Waypoints at `[0, 5, 10, 15, 19]` for `T=20, N=3`. Zero within-sample variance, matches the closest paper analog.
2. **Bounded random** — keep random but enforce `min_gap` (say 3 frames) so segments can't degenerate to length 1.

I'm leaning option 1 (fixed stride) — same spec PLDM used in HWM, and removes a tunable confound when comparing our success rate against the paper's 87% on TwoRoom.

## Code pointers

- `sample_waypoints` now lives in `waypoint_sampler.py:41` (moved out of `hierarchical_lewm.py`). The same file also exports `sample_waypoints_fixed_stride` (PLDM-style, line 71) and `sample_waypoints_variable_span` (DINO-WM-style, line 110) so the fix is now a one-line import swap inside `train_hierarchical_lewm`.
- Called once per batch from `train_hierarchical_lewm`, `hierarchical_lewm.py:552` — so every trajectory in a batch shares the same waypoint indices (separate but related side-effect: no padding mask is ever needed inside `A_psi`).

---

## Update — confirmed against the HWM paper (arXiv 2604.03208)

Pulled the HWM PDF and read the per-backbone training setups. The conclusion: **our sampler has no precedent in the paper**, and the closest analog (PLDM) is fully deterministic.

### What the paper claims in general (§2.3, page 4)

> "Unlike fixed-stride temporal abstraction methods, we do not assume a fixed high-level horizon h, allowing each high-level transition to correspond to a variable-length segment of low-level execution."

OK so in principle HWM embraces variable spans. But each backbone implements this very differently:

### What each backbone actually does

**VJEPA2-AC (Franka) — Appendix B.2:**
> "We use N = 3 waypoint states sampled from trajectory segments spanning up to 4 seconds, with the middle waypoint chosen uniformly at random."
>
> "...trajectory segments {(a_k, p_k, x_k)}_{k∈T} spanning (0.33, 4) seconds and select N = 3 waypoint indices per segment, with the middle waypoint sampled uniformly at random."

Total span per sample: `Uniform(0.33s, 4s)`. Only 2 segments per sample (N=3 → 2 endpoints + 1 interior). Within-sample ratio bounded but not tight.

**DINO-WM (Push-T) — Appendix B.3:**
> "To construct training sequences, we subsample trajectory segments with lengths uniformly drawn between 25 and 70 timesteps. From each segment, we sample N = 5 waypoint states, which define the high-level transitions."

Total span per sample: `Uniform(25, 70)` timesteps. N=5 waypoints, 4 segments. How the 5 waypoints are placed within the segment is not specified.

**PLDM (Diverse Maze, the closest analog to LeWM) — Appendix B.4:**
> "To construct training sequences, we subsample 60 timesteps from each trajectory and extract 6 waypoint states using a fixed stride of 10 timesteps."

Total span: deterministic 60 timesteps. Waypoints at `[0, 10, 20, 30, 40, 50]`. 5 segments of exactly 10 timesteps each. **Zero within-sample variance.** Directly contradicts §2.3's "variable-length" framing — for the JEPA-from-pixels backbone, they reverted to fixed stride.

### Comparison to our code

| Setup | Total span per sample | N waypoints | # segments | Within-sample variance |
|---|---|---|---|---|
| **Our code** (T=20, N=3) | fixed (20 frames = 100 raw steps) | 5 (3 interior + 2 endpoints) | 4 | **unbounded** (pathological 16× possible) |
| PLDM (closest analog) | fixed 60 raw steps | 6 | 5 | **zero** (fixed stride 10) |
| DINO-WM Push-T | `Uniform(25, 70)` | 5 | 4 | bounded; placement unstated |
| VJEPA2-AC Franka | `Uniform(0.33s, 4s)` | 3 | 2 | bounded; middle ∈ uniform |

We are inconsistent with the closest backbone (PLDM) in two ways:
1. **Random interior indices instead of fixed stride.** PLDM uses deterministic stride 10; we use `randperm` with no `min_gap`.
2. **Random waypoint placement inside a fixed-span window.** That specific combo does not appear in any of the three HWM backbone setups.

The paper also does not discuss padding for variable-length action chunks — the transformer + CLS handles it architecturally, but it's not addressed because PLDM has constant-length chunks and DINO-WM / VJEPA2-AC vary the *total* span across samples rather than within.

### Recommendation (unchanged from above)

Switch to **fixed stride** waypoints, matching PLDM exactly. For T=20, N=3 that's `[0, 5, 10, 15, 19]` (or `[0, 4, 9, 14, 19]` for exact stride-5). This removes a confound when comparing our TwoRoom success rate to the paper's 87%.

### What about variable episode lengths?

The paper doesn't discuss episodes too short for the minimum span. Their datasets are sized to avoid the problem (VJEPA2-AC = real robot, DINO-WM = 18.5k trajectories, PLDM = synthetic 100-step episodes). For us this matters: with `span=100`, only 6,165 / 10,000 TwoRoom episodes are long enough — a 38% drop. Not changeable without shortening the span.

### Resolution note (2026-05-27)

We dropped to `stage2_num_steps=12` (span = 12 × frameskip(5) = **60 raw env steps**), matching the paper exactly. At span=60, **670,809 valid starting points** are found in the dataset (vs 6,165 at the original span=100) — episode-length drop is essentially eliminated.

---

# Issue 2: inner CEM iterations were too few — **biggest cause of the 20 % ceiling** (FOUND + RESOLVED 2026-05-27)

## TL;DR

`hierarchical_plan.cem` was being run with **only 5 refinement iterations** per planner call. The inner planner searches a **100-dimensional** action space (`h_low=10` effective steps × `action_dim=10`). CEM in that dimensionality typically needs **20–50 iterations** to converge. Flat LeWM's reference solver (`stable_worldmodel.solver.CEMSolver`) uses `n_steps=30`.

Raising `plan.inner_iters: 5 → 30` (CLI override, no retraining, no code change) moved success rate **36 % → 62 %** on TwoRoom. Run `bdi0rrzam` in `RUNS.md`.

## How we found it

1. After applying training-side + eval-side fixes (run `bgb6k93mg`), we landed at 36 % — better than the 20 % baseline but far below flat LeWM's 92 %.
2. The cross-repo investigation had pointed at the macro-action distribution (`A_ψ` outputs not matching the planner's `N(0,1)` sampling prior) as the dominant cause (TIER 1 in the diagnostic report). Predicted fix: replace `A_ψ` with `actions.sum + normalize`, retrain.
3. Sanity baseline test (`bwt2z54mj`) — patch `z_sg = z_goal` in `hierarchical_plan.py:152`, bypassing the outer planner entirely. **Expected:** if the macro-action mismatch is the issue, this should jump to ~80–90 %.
4. **Observed:** sanity baseline gave **32 %**, indistinguishable from the 36 % full run. So the outer planner + `A_ψ` + `P²` are *not* the dominant bottleneck. **The inner planner can't reach a faraway subgoal even when handed the true goal.**
5. Comparing our inner CEM to flat LeWM's `CEMSolver`:

   | knob | ours | flat LeWM | ratio |
   |---|---|---|---|
   | `n_iters` | 5 | 30 | 6× |
   | `num_samples` | 256 | 300 | ≈ |
   | warm-start across replans | no | (probably yes — needs verifying) | — |

6. Bumped `plan.inner_iters=30` as a CLI override (no code change, no retrain). Run `bdi0rrzam` → **62 % (31/50)**, 95 % CI [48 %, 75 %]. CIs do not overlap with the 36 % run. Confirmed.

## Why 5 iters was the original default

This is the planner reimplemented in `hierarchical_plan.py:cem` for the H-LeWM extension; it does not inherit from `stable_worldmodel.solver.CEMSolver`. The 5/5 defaults likely came from "matches the outer planner's iteration count" without checking the action-space dimensionality the inner planner has to optimize over. Outer is 12-dim (`H_high * d_L = 3 * 4`); inner is 100-dim — they aren't symmetric.

## Recommended action

1. **Update the eval config default**: `plan.inner_iters: 5 → 30` in `config/eval/hierarchical.yaml`. One-line change.
2. **Optional follow-up**: also bump `plan.inner_samples: 256 → 300` to fully match flat's `CEMSolver`.
3. **Bigger follow-up (not yet tested)**: add warm-start to `cem()` so `mu` and `std` persist across MPC replans. Flat's `CEMSolver` likely does this; our `cem()` re-initializes from zero on every `plan()` call.

## What this *doesn't* fix

The 30-point gap to flat LeWM (62 % vs 92 %) is still open. After the inner-iters bump, the remaining gap is plausibly:

- **CEM warm-start across replans** (above) — cheap structural test
- **Cost function** — both ours and flat use endpoint L1; both probably benefit from cumulative cost
- **MPPI vs CEM** — paper PLDM uses MPPI; our `cem()` is unrelated to the paper's planner
- **Rollout loss vs teacher-forced training** — paper PLDM trains pure rollout (`γ_tf=0, γ_roll=1, predT=6`); we train pure teacher-forced L1. Bigger change, requires retraining.

See `CONTEXT.md` §8 for the prioritized list.
