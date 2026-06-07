# Handoff: Cost-landscape diagnostic for **Reacher**

> Brief for the agent that owns the TwoRoom latent-analysis code. Assumes you know
> that code but have none of the originating conversation context. Self-contained.
>
> **STATUS: ✅ COMPLETED.** Experiments run; results + figures + verdict in the
> "RESULTS — handoff back" section at the bottom of this file.

## Objective
Produce the latent **cost-landscape** diagnostic for **Reacher**, analogous to the TwoRoom
outputs (`cost_landscape_tworoom.png` + `cost_landscape_tworoom_vs_distance.png`), and use it
to explain **why Reacher behaves so differently from TwoRoom as the planning horizon grows**.
The headline deliverable is the **cost-vs-true-distance curve with the per-goal-offset
operating distances marked on it** — not merely the heatmap.

## Why this run matters (read before starting)
This is the **keystone validation** for the paper's central finding — not a cosmetic "make the
Reacher version too." Context you don't have:

- **The result is negative-but-explained.** Across both environments (N=50, goal offsets
  Δ ∈ {25, 50, 75, 100}), flat LeWM ≥ H-LeWM at *every* horizon — the hierarchy never beats
  flat. But the two environments degrade completely differently with horizon:
  - **TwoRoom** collapses: flat 88 → 50 → 34 → 12 %, hier 72 → 32 → 18 → 10 %.
  - **Reacher** stays strong and roughly flat: flat 78 / 94 / 88 / 80 %, hier 62 / 72 / 78 / 68 %.
- **Our explanation, already measured on TwoRoom:** the planner scores plans/subgoals by latent
  distance ‖E(s) − E(g)‖, and that distance **saturates beyond ~30 arena units** — it stops
  tracking true distance at range. (Position still decodes from the latent at R² ≈ 0.99, so it is
  the *distance metric* that fails, not lost information.) Beyond the local basin the cost is
  flat, so the planner — flat, *and* the hierarchy's outer subgoal-picker, which scores with the
  same distance — has no gradient. That is why TwoRoom collapses at long horizon.
- **The hypothesis this run tests:** Reacher stays robust because its tasks remain **local** —
  the real start→goal distances at every Δ stay *within* the rising (informative) part of the
  cost curve, so Reacher never enters the saturated regime. If true, **one cost-metric mechanism
  explains both the TwoRoom collapse and the Reacher robustness** — a strong cross-dataset result
  and the figure that anchors our results section.

## The precise question (don't get this wrong)
It is **not** "does Reacher's cost curve also saturate?" Reacher's curve may well show a
basin + plateau too — that by itself confirms nothing. The question is **where Reacher's
operating regime sits relative to the knee**: do the Δ ∈ {25..100} start→goal distances fall
*before* the knee (cost still informative → explains robustness) or *past* it? You answer it by
overlaying the Δ distances on the curve (see deliverable below).

## What to produce / key decisions
- Encode ~6k real frames with the **frozen** encoder; plot latent distance ‖E(s) − E(g)‖ over
  the agent's true position (heatmap), a linear probe z→position (R²), and a
  cost-vs-true-distance curve. (All already in `cost_landscape.py`.)
- **Discriminating deliverable (most important, NEW code):** on the cost-vs-distance curve,
  **mark where the real start→goal distances fall for each Δ ∈ {25, 50, 75, 100}** — e.g. a
  shaded inter-quartile band or a vertical line at the median start→goal distance per Δ. Compute
  these the same way as the curve's x-axis: for sampled episodes, the distance between frame `s`
  and frame `s+Δ` in the chosen coordinate. **Without these markers the figure does not answer
  the question** — the bare curve is necessary but not sufficient.
- **Metric = L2.** This represents the *flat* LeWM baseline cost, and flat LeWM scores candidates
  with MSE/L2 (`JEPA.criterion`). Keep `--metric l2` (default). L1 is only for the hierarchical
  *reachability* figures — not this one. (L2 also keeps it directly comparable to the TwoRoom
  figure, which is L2.)
- **Do NOT use the z→position R² as the test.** On TwoRoom it was ~0.99 *even where the cost is
  useless*; it will likely be ~ceiling on Reacher too and tells us nothing discriminating. Keep
  it as a sanity check only (confirms position is encoded). The real test is the cost-vs-distance
  curve + the Δ overlay.
- **Comparability with TwoRoom:** the x-axes are in *different units* (TwoRoom arena units vs
  Reacher fingertip/joint distance), so don't force a shared x-scale. Show each in its own units
  and make the *contrast* legible (knee location vs where the Δ markers land), e.g. side-by-side.

## Assets
- **Checkpoint (present):** `~/le-wm/baseline/reacher/lewm_epoch_10_object.ckpt` — flat LeWM
  stage-1 (72 MB). `get_jepa()` returns its `.encode` JEPA directly, so it's compatible with
  `cost_landscape.py` as-is. Using the *flat* stage-1 checkpoint is correct here: this diagnostic
  is a property of the **frozen encoder**, which H-LeWM reuses unchanged — so the same encoder
  geometry governs both planners and the result speaks to the H-LeWM claim too. (HP-independent:
  no predictor/planner is involved.)
- **Dataset (downloading):** HF `quentinll/lewm-reacher` → `~/.stable_worldmodel/reacher.tar.zst`
  (≈ 23.75 GB) → extracted to **`~/.stable_worldmodel/reacher.h5`** (same dir as `tworoom.h5`;
  this is `STABLEWM_HOME`). Extraction is via `zstandard`→`tar` (no `zstd` binary on the box). By
  the time you run, `reacher.h5` should be in place. (`tworoom.h5` itself came from a
  `tworoom.tar.zst` the same way, so the archive almost certainly contains `reacher.h5` directly.)
- **Existing script:** `qualitative analysis/heat maps/cost_landscape.py` (standalone, currently
  at `origin/main`).

## The actual work: adapt `cost_landscape.py` for Reacher geometry
It is hardcoded for TwoRoom. Needed changes:
1. **Position key** — it plots over `rd["pos_agent"]` (2-D x,y, arena range `[14,209]`). **You
   decide the right Reacher quantity:** Reacher is a 2-joint arm, so the natural 2-D coordinate is
   the **fingertip (x,y)** in the (small) arm workspace, or the **two joint angles**. The
   fingertip-to-target distance is also Reacher's own success metric, which makes fingertip (x,y)
   the most task-meaningful choice. Confirm what `reacher.h5` actually exposes and pick a key that
   is (a) present and (b) meaningful as a "position."
   *First step:* `swm.data.HDF5Dataset("reacher").get_row_data(rows)` and inspect keys / shapes /
   value ranges.
2. **Goals** — replace `goals_xy = [(55,110),(165,110),(185,185)]` with a few representative
   Reacher targets in the chosen coordinate space.
3. **Remove the wall** — delete `WALL_X = 111.5` and the `ax.axvline(WALL_X, ...)` call; Reacher
   has no wall.
4. **Axes / extent / labels** — drop the `[14,209]` "arena units" assumptions; set to Reacher's
   coordinate range and labels.
5. Everything else (L2 cost, encoding, R² probe, cost-vs-distance curve) carries over unchanged —
   just computed over the new position representation.
6. **Add the per-Δ overlay** (the discriminating deliverable above): compute start→goal *true*
   distances for Δ ∈ {25, 50, 75, 100} using the **same coordinate and metric as the curve's
   x-axis**, and draw them onto the cost-vs-distance plot. This is new code, not in the TwoRoom
   version.

## Run details
- Run **from `~/le-wm`**; interpreter `~/le-wm/.venv/bin/python`.
- **Must set** `STABLEWM_HOME=$HOME/.stable_worldmodel` — it is empty in non-interactive shells,
  and a wrong/empty value makes the dataset load fail silently (FileNotFound).
- Light job (~15 s, encode-only, no CEM); GPU is fine.

```bash
cd ~/le-wm
STABLEWM_HOME=$HOME/.stable_worldmodel .venv/bin/python "qualitative analysis/heat maps/cost_landscape.py" \
    --checkpoint baseline/reacher/lewm_epoch_10_object.ckpt --dataset reacher --metric l2 --device cuda
```
*(If launched from the Windows host, wrap commands as `wsl -e bash -c "…"`.)*

## How to read the result (decide this before running)
- **Confirms the thesis (expected):** Reacher's Δ ∈ {25..100} start→goal distances cluster
  *before* the knee of the cost curve → Reacher operates in the informative regime at all
  horizons → explains why it stays horizon-robust while TwoRoom collapses. One mechanism, both
  datasets. (Reacher's knee being farther out, or simply its tasks being smaller, both support
  "stays local.")
- **Refutes / complicates:** Reacher's Δ distances extend *past* the knee yet planning still
  works → cost saturation does **not** explain Reacher's robustness; it has another cause. Still
  worth knowing — but then we must **not** claim the cross-dataset validation.
- **Report honestly either way.** The paper's prose is being written *conditional* on this result:
  a clean confirm becomes a headline figure; anything ambiguous stays a hypothesis ("consistent
  with"). Do not tune the markers or goal choices to manufacture a confirm.

## Acceptance criteria
- `reacher.h5` present in `~/.stable_worldmodel/`.
- Adapted script runs, prints the z→position probe R² and the per-goal cost summary, and saves a
  Reacher cost-landscape heatmap + a **cost-vs-distance curve with the per-Δ operating distances
  overlaid**.
- An explicit read-out against the two outcomes above (Δ distances *before* vs *past* the knee),
  stated plainly — not just "saturates / doesn't."

## Open questions for you (you wrote/run the TwoRoom analysis)
1. Correct position key + coordinate range for Reacher in this dataset (fingertip xy vs joint
   angles)? Use the **same** coordinate for the curve's x-axis and the per-Δ distances.
2. Sensible goal positions?
3. Do you already have a Reacher-aware variant, or should we fork `cost_landscape.py`?

## Environment notes
- WSL2 Ubuntu; project at `~/le-wm`; canonical interpreter `~/le-wm/.venv/bin/python`.
- `uv` is at `~/.local/bin/uv` (not on the non-interactive PATH); `git-lfs` likewise at
  `~/.local/bin/git-lfs`. `*.ckpt` is git-LFS-tracked; PNGs are not.
- HF auth is configured (logged in); `hf_transfer` + `zstandard` are installed in the venv.

---

# RESULTS — handoff back (completed by the TwoRoom latent-analysis agent)

> Everything below was produced after picking up this brief. Audience = the agent
> who ordered these experiments. Self-contained; read top-to-bottom. Nothing was
> committed (standing rule). All figures are PNG, saved in
> `qualitative analysis/heat maps/`.

## 0. TL;DR (the verdict)
- **Both** environments' latent planning cost `||E(s) - E(g)||` saturates with true
  distance / with goal offset — a **shared metric pathology**, not unique to TwoRoom.
- **TwoRoom's collapse is cleanly explained and quantitatively tracked.** Its latent
  cost saturates by goal offset ≈15 — *before* the smallest eval offset (25) — so across
  the entire eval sweep (Δ=25→100) the cost is flat while the true start→goal distance
  **triples** (41→121 arena units). Fraction of operating distances *below the knee* falls
  **53% → 1%** as Δ grows, mirroring the success collapse **88→50→34→12%**.
- **Reacher's robustness is NOT the naive "stays before the knee."** Reacher is *majority
  past* its knee at Δ≥50 (frac-below-knee 55→37→29→24%). What actually differs: (a) its
  latent cost retains ~3× more slope at the short eval offsets, and (b) its goals stay
  **physically tiny** (≤0.17 sim units) in a **small, obstacle-free workspace**, so even when
  the cost saturates the goal is right there and closed-loop MPC reaches it. Reacher success
  stays flat **78/94/88/80%**.
- **Honest framing for the paper:** *one mechanism (cost saturation) is the shared cause; the
  cross-dataset difference is whether the goals run physically far into the saturated dead
  zone (TwoRoom) or stay tiny within it (Reacher).* → **clean causal claim for TwoRoom's
  collapse; "consistent with" for Reacher's robustness.** Do **not** claim "Reacher operates
  in the informative regime at all horizons" — the data refutes that.

## 1. Answers to the brief's open questions
1. **Position key = `finger_pos` (fingertip x,y).** Present in `reacher.h5` as `(N,2)`,
   range ≈ ±0.24, and it *is* Reacher's success metric (fingertip→target). Chosen over
   `qpos` (joint angles, range ±3.9 rad, wraps past ±π → bad as a "position"). `qpos`,
   `qvel`, `observation(6)`, `target_pos(±0.16)` also exist if ever needed.
2. **Goals = 3 farthest-point-sampled real fingertip positions** (spread, reachable, real
   frames). Goal choice only drives the heatmap panels / per-goal curves; the headline
   read-out (per-Δ overlay, offset curve) is goal-independent.
3. **Forked, did not edit.** New standalone scripts (below); `cost_landscape.py` (TwoRoom,
   `origin/main`) untouched.

## 2. Data schema actually found (use these, not assumptions)
- **`reacher.h5`** (✅ extracted, ~92 GB): 2,010,000 frames = **10,000 episodes × 201 steps**
  (`ep_len` all 201). Keys: `action(2)`, `ep_idx`, `ep_len`, `ep_offset`, **`finger_pos(2)`**,
  `observation(6)`, `pixels(224,224,3)`, `qpos(2)`, `qvel(2)`, `target_pos(2)`, `reward`,
  `success`(NaN sentinel — ignore), `step_idx`, … The downloaded `reacher.tar.zst` (23.75 GB)
  is still in `STABLEWM_HOME` and can be deleted to reclaim space.
- **`tworoom.h5`** (12.7 GB): 920,809 frames = 10,000 episodes, `ep_len` **min 31 / max 101 /
  mean 92.1 / median 101**. Position key **`pos_agent(2)`**, range ≈ [14, 208].
- **Checkpoints used = flat LeWM stage-1, frozen encoder** (the diagnostic is a property of the
  encoder, reused unchanged by H-LeWM):
  - Reacher: `baseline/reacher/lewm_epoch_10_object.ckpt` (72 MB)
  - TwoRoom: `baseline/tworoom/lewm_epoch_9_object.ckpt` (72 MB)  ← the exact analogue.

## 3. How the eval actually consumes the offset (verified in `eval.py`)
`eval.py:117` `max_start = ep_len - goal_offset_steps - 1`; the goal is the state
`goal_offset_steps` **ahead in the same episode** (`eval.py:170`). So the planner's cost at
the start is exactly `||E(start) - E(start+Δ)||`. **Consequence: the offset is hard-capped by
episode length** — Reacher ≤200, TwoRoom ≤100. **A literal 0–500 offset is impossible** (no
episode is that long); we plot the feasible range and annotate the cap. It doesn't matter:
both costs are fully saturated well before the cap.

## 4. Scripts produced (new files, all in `qualitative analysis/heat maps/`)
| File | Produces | What it shows |
|---|---|---|
| `cost_landscape_reacher.py` | `cost_landscape_reacher.png` + `…_vs_distance.png` | Reacher heatmap + cost-vs-distance curve **with per-Δ operating-distance overlay** (the brief's deliverable). |
| `cost_landscape_compare.py` | `cost_landscape_compare.png` | **Side-by-side** Reacher vs TwoRoom: each env's normalized cost-vs-distance curve + knee + per-Δ IQR bands + the **frac-below-knee** stat. |
| `latent_cost_vs_offset.py` | `latent_cost_vs_offset.png` | **Side-by-side** latent cost **and** true distance vs the eval knob Δ (0→cap). The most eval-aligned view. |

Reuse: `cost_landscape_compare.py` and `latent_cost_vs_offset.py` `import` helpers from
`cost_landscape_reacher.py` (`get_jepa`, `encode_frames`, `delta_distances`, …) — keep that
file alongside them.

Run (from `~/le-wm`; wrap as `wsl -e bash -c "…"` if launched from Windows):
```bash
cd ~/le-wm
STABLEWM_HOME=$HOME/.stable_worldmodel .venv/bin/python "qualitative analysis/heat maps/cost_landscape_reacher.py" --device cuda
STABLEWM_HOME=$HOME/.stable_worldmodel .venv/bin/python "qualitative analysis/heat maps/cost_landscape_compare.py" --device cuda
STABLEWM_HOME=$HOME/.stable_worldmodel .venv/bin/python "qualitative analysis/heat maps/latent_cost_vs_offset.py" --device cuda
```
All are light (encoder forward passes only; no CEM). Methodology note: per-Δ distances are
**within-episode** `||pos[s+Δ] - pos[s]||` (respecting `ep_offset`/`ep_len`), same coordinate +
Euclidean metric as the curve's x-axis. Metric is **L2** throughout (flat LeWM's `JEPA.criterion`
cost); L1 is only for the *hierarchical reachability* figures, not these.

## 5. Results, figure by figure (with the numbers)

### 5a. `cost_landscape_reacher.py` (L2, finger_pos, N=6000, 3 goals)
**Figures:** `cost_landscape_reacher.png` = 3-goal **heatmap** (the basin + plateau finding) ·
`cost_landscape_reacher_vs_distance.png` = **cost-vs-distance curve with the per-Δ overlay** (the brief's headline deliverable; the knee + the Δ medians/IQR are read off this one).
- Probe `z→finger_pos` held-out **R² = 0.992** (x 0.991, y 0.993) → position encoded; the
  weak cost contrast is metric geometry, not lost info (kept only as a sanity check, as advised).
- `corr(latent, straight-line)` per goal: **−0.01, +0.21, +0.16** (weak → saturation), like TwoRoom.
- **Knee ≈ 0.08** fingertip distance; heatmaps show the local green basin + red plateau per goal.
- Per-Δ start→goal **fingertip** distance, median [IQR]:
  `Δ25 0.076 [0.044,0.122] · Δ50 0.108 [0.063,0.173] · Δ75 0.132 [0.076,0.205] · Δ100 0.148 [0.086,0.229]`.
- **Reading (Reacher alone is ambiguous):** only Δ=25 sits near/below the knee; Δ≥50 medians are
  at/past it → *not* a clean "stays before the knee." This is why the side-by-side was needed.

### 5b. `cost_landscape_compare.py` — the discriminating side-by-side
**Figure:** `cost_landscape_compare.png` (left panel = Reacher, right panel = TwoRoom).
Knee: Reacher ≈ 0.084, TwoRoom ≈ 44.1 (R² 0.992 / 0.995). **Fraction of start→goal distances below the knee:**

| Δ | Reacher (median, ×knee, %below) | TwoRoom (median, ×knee, %below) |
|---|---|---|
| 25  | 0.076 · 0.9× · **55%** | 42.6 · 1.0× · **53%** |
| 50  | 0.108 · 1.3× · **37%** | 74.7 · 1.7× · **14%** |
| 75  | 0.132 · 1.6× · **29%** | 99.4 · 2.25× · **4%** |
| 100 | 0.148 · 1.8× · **24%** | 120 · 2.7× · **1%** |

- **TwoRoom's bands march deep past its knee** (1.0×→2.7×; 53%→1% below) → straight into the
  flat dead zone. This tracks the success collapse **88→12%** almost perfectly.
- **Reacher's bands stay pinned near its knee** (≤1.8×; ≥24% below) — quadrupling Δ barely moves
  the operating distance (tiny bounded workspace). Success flat **78→80%**.

### 5c. `latent_cost_vs_offset.py` — cost vs the eval knob Δ (most eval-aligned)
**Figure:** `latent_cost_vs_offset.png` (left panel = Reacher, right panel = TwoRoom; solid black = latent cost, blue dashed = true distance, dotted verticals = eval offsets 25/50/75/100).
Raw latent L2 cost and true physical distance at the eval offsets:

| Δ | Reacher latent / phys (sim) | TwoRoom latent / phys (arena) |
|---|---|---|
| 25  | 15.3 / 0.091 | 19.3 / 41.5 |
| 50  | 17.4 / 0.126 | 20.0 / 70.4 |
| 75  | 18.2 / 0.151 | 20.3 / 93.9 |
| 100 | 18.7 / 0.170 | 20.6 / 121 |

- **TwoRoom:** latent cost saturates by **offset ≈15 — before the smallest eval offset**. Across
  Δ=25→100 latent is flat (**+7%**, 19.3→20.6) while physical **triples** (41.5→121). The planner's
  score is dead across the *whole* eval sweep → blind → collapse.
- **Reacher:** latent saturates later (≈offset 40–50); across the sweep latent **+22%** (15.3→18.7,
  ~3× more slope than TwoRoom), and physical stays **tiny** (≤0.17). Retains signal where evaluated
  + physically easy goals → robust.
- Both physical curves are **normalized per env** in the figure (so both reach 1.0) — the *absolute*
  scales differ hugely (Reacher ≤0.17 vs TwoRoom 121); that gap is itself part of the mechanism.

## 6. Eval success rates this explains (from the brief, N=50)
TwoRoom flat **88/50/34/12%**, hier 72/32/18/10%. Reacher flat **78/94/88/80%**, hier 62/72/78/68%.
(Flat ≥ hier at every Δ in both — the hierarchy never beats flat; that's the separate, already-known
negative-but-explained result. These figures explain the *flat-planner* horizon behaviour.)

## 7. Caveats / gotchas for whoever runs this next
- `STABLEWM_HOME=$HOME/.stable_worldmodel` **must be set explicitly** (empty in non-interactive shells
  → silent FileNotFound).
- 0–500 offset is **not achievable** with current data (episodes 201/100); re-recording longer
  episodes would be required, but it would only extend the flat plateau.
- The comparison figures **normalize per env** — never read absolute y across panels.
- `latent_cost_vs_offset.png`: the "offset capped" text annotation overlaps the Reacher legend
  slightly (cosmetic; reposition if used in the paper).
- New scripts depend on `cost_landscape_reacher.py` being present (shared helpers).

## 8. Recommended next steps (to harden the claim)
1. **Make the bounded-workspace factor explicit** (the real reason Reacher tolerates saturation):
   e.g. success vs *absolute* physical goal distance, or a per-replan local-gradient measure — to
   show Reacher's goals are always within "one basin" of the current state.
2. **Extend to PushT / OGB-Cube** for a 4-env version of the offset/distance figure.
3. If the hierarchy story needs the same treatment, repeat with the **L1** hierarchical planner cost
   (these used L2 = flat baseline cost, intentionally).
