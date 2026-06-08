# How the TwoRoom dataset is actually laid out, step by step

Ground-up explanation using actual numbers pulled from `~/.stable_worldmodel/tworoom.h5`.

## 1. The file is a flat row-store, NOT one-row-per-episode

There are **920,809 rows** in the file. **Each row is one raw environment timestep** — i.e., one tick of the TwoRoom physics simulator (one action, one image, one set of positions). There are no nested groups for episodes — episodes are just contiguous runs of rows.

10,000 episodes total. They have variable length (31–101 raw steps, mean 92.1, median 101). They're stored back-to-back:

```
row  0 ........... row 60  ← episode 0 (61 rows)
row 61 ........... row 138 ← episode 1 (78 rows)
row 139 .......... row 219 ← episode 2 (81 rows)
...
row 920708 ....... row 920808 ← episode 9999 (101 rows)
```

Two helper arrays tell you where episodes start and how long they are:

| Array | Shape | Meaning |
|---|---|---|
| `ep_offset` | `(10000,)` int64 | row index where each episode begins |
| `ep_len` | `(10000,)` int32 | length of each episode (in raw rows) |
| `ep_idx` | `(920809,)` int32 | for each raw row, which episode it belongs to |
| `step_idx` | `(920809,)` int64 | for each raw row, the step number within its episode (0..ep_len-1) |

The first three arrays are redundant — `ep_idx` is computable from `ep_offset/ep_len`. The `swm.data.HDF5Dataset` uses `ep_offset/ep_len` for slicing.

## 2. The columns — what each row actually contains

| Column | Shape per file | Per-row | Meaning |
|---|---|---|---|
| `pixels` | `(920809, 224, 224, 3)` uint8 | 224×224 RGB image | the raw observation (138 GB nominal, ~12 GB on disk via HDF5 compression) |
| `action` | `(920809, 2)` float32 | 2D vector ∈ [-1, 1] | the primitive action (`dx`, `dy` for the navigator) |
| `proprio` | `(920809, 2)` float32 | (x, y) position | agent's own position — what the eval harness uses as state, since TwoRoom has no separate `state` column |
| `pos_agent` | `(920809, 2)` float32 | (x, y) | same as `proprio` for TwoRoom |
| `pos_target` | `(920809, 2)` float32 | (x, y) | the target/goal position |
| `distance_to_target` | `(920809,)` float64 | scalar | Euclidean distance, used as the success criterion |
| `observation` | `(920809, 10)` float64 | 10-d vector | a low-d state vector from the env (unused by LeWM training, which uses pixels) |
| `reward`, `terminated`, `truncated` | scalars | standard gym RL bookkeeping |

For LeWM/H-LeWM training we only care about `pixels`, `action`, and `proprio`. The others are there for evaluation (the env needs to know agent + target positions to set up states/goals).

## 3. The two knobs that turn rows into training samples: `frameskip` and `num_steps`

This is the part that trips people up. The dataset doesn't hand you one row per training step. It hands you **a small window of consecutive rows, downsampled**.

Two config knobs:

- **`frameskip = 5`** — only keep every 5th frame in the window for pixels/proprio. The 5 raw actions in between get **packed into one "effective action"** of size `5 × 2 = 10`. So the model sees a coarser-grained trajectory.
- **`num_steps`** — how many *frames* (after frameskip downsampling) each sample contains.
- These multiply to define the **span**: `span = frameskip × num_steps` = how many consecutive raw rows are needed to produce one sample.

|  | Stage 1 (`train.py`) | Stage 2 (`train_hierarchical.py`) |
|---|---|---|
| `frameskip` | 5 | 5 |
| `num_steps` | 4 | 20 |
| `span` (raw rows) | 20 | 100 |
| Effective action dim | 10 | 10 |

Then `HDF5Dataset.__init__` enumerates every legal `(episode, start_offset)` such that `start_offset + span <= ep_len`:

```python
# stable_worldmodel/data/dataset.py:44-49
self.clip_indices = [
    (ep, start)
    for ep, length in enumerate(lengths)
    if length >= self.span
    for start in range(length - self.span + 1)
]
```

That's how many distinct training windows are available:

| Setting | Span | Episodes with `length >= span` | Total clips |
|---|---|---|---|
| Stage-1 (`span=20`) | 20 | 10000 / 10000 | **730,809** |
| **Stage-2 (`span=100`)** | **100** | **6,165 / 10,000** | **12,221** |

**Stage-2 has 60× less training data than Stage-1**, simply because most TwoRoom episodes are too short to fit a 100-step window. That's a real consideration for training Stage-2.

## 4. What `__getitem__` actually returns

When the dataloader calls `dataset[idx]`, here's what happens (see `HDF5Dataset._load_slice` in `.venv/lib/python3.10/site-packages/stable_worldmodel/data/dataset.py`):

1. Look up `(ep_idx, start)` from `clip_indices[idx]`.
2. Compute absolute raw-row range `[g_start, g_end) = [ep_offset[ep] + start, ep_offset[ep] + start + span)`.
3. For each column, slice those `span` raw rows.
4. For all columns **except `action`**: downsample with `[::frameskip]`. So pixels/proprio/etc. go from `span=100` rows to `num_steps=20` frames.
5. For `action`: keep all 100 raw rows, then reshape to `(num_steps=20, frameskip*action_dim=10)`. So action `[k, :]` packs the 5 raw actions executed between frame `k` and frame `k+1`.

The result for one Stage-2 batch element on TwoRoom:

```text
pixels  : (20, 3, 224, 224)  uint8   -- 20 images, sampled every 5 raw steps
action  : (20, 10)           float32 -- 20 effective actions, each a stack of 5 raw 2D actions
proprio : (20, 2)            float32 -- agent (x,y) at each of those 20 frames
```

After image preprocessing and column normalization in `train_hierarchical.py`, the dataloader collates `B` of these into:

```text
batch["pixels"]  : (B, 20, 3, 224, 224)  float32  (ImageNet-normalized)
batch["action"]  : (B, 20, 10)           float32  (StandardScaler-normalized, NaN-replaced)
batch["proprio"] : (B, 20, 2)            float32  (StandardScaler-normalized)
```

## 5. A concrete walk-through: episode 6 of TwoRoom

Episode 6 has length 101 (long enough for stage-2), starts at raw row 501. Suppose the dataloader pulls `(ep=6, start=0)`, so it loads raw rows `[501, 601)`. After `[::5]` downsampling, the 20 frame indices are:

```
frame  0 = raw row 501 ; agent at (183.3, 74.4)   ← top-right room?
frame  1 = raw row 506 ; agent at (176.7, 67.7)
frame  2 = raw row 511 ; agent at (176.9, 61.4)
frame  3 = raw row 516 ; agent at (161.9, 58.1)
frame  4 = raw row 521 ; agent at (148.7, 59.3)
frame  5 = raw row 526 ; agent at (143.6, 63.1)
frame  6 = raw row 531 ; agent at (136.7, 54.4)
frame  7 = raw row 536 ; agent at (119.7, 48.0)
frame  8 = raw row 541 ; agent at (128.7, 73.0)   ← jumped — through a doorway?
frame  9 = raw row 546 ; agent at (119.5, 62.3)
frame 10 = raw row 551 ; agent at (115.9, 50.9)
frame 11 = raw row 556 ; agent at ( 96.0, 50.1)
frame 12 = raw row 561 ; agent at ( 96.5, 38.6)
frame 13 = raw row 566 ; agent at ( 97.9, 53.4)
frame 14 = raw row 571 ; agent at ( 93.7, 59.0)
frame 15 = raw row 576 ; agent at ( 90.5, 73.1)
frame 16 = raw row 581 ; agent at ( 71.3, 77.0)   ← crossing into the other room
frame 17 = raw row 586 ; agent at ( 79.7, 70.2)
frame 18 = raw row 591 ; agent at ( 76.4, 67.2)
frame 19 = raw row 596 ; agent at ( 70.8, 75.7)   ← ended near goal
```

You can see this is a *real spatial trajectory* — the agent moves from (183, 74) over to about (70, 76), wandering between rooms. The episode is roughly 100 raw env steps long and we've summarized it as 20 frames spaced 5 raw steps apart.

## 6. NOW `sample_waypoints` makes sense

For this one episode-6 sample, [`sample_waypoints(T=20, N=3)`](../hierarchical_lewm.py:59) might return `[0, 4, 11, 16, 19]`. Mapped onto the actual trajectory:

| Waypoint | Frame | Agent position | Raw env step |
|---|---|---|---|
| `z_0`     | frame 0  | (183.3, 74.4) | step 0 |
| `z_1`     | frame 4  | (148.7, 59.3) | step 20 |
| `z_2`     | frame 11 | (96.0, 50.1)  | step 55 |
| `z_3`     | frame 16 | (71.3, 77.0)  | step 80 |
| `z_4`     | frame 19 | (70.8, 75.7)  | step 95 |

So the four segments between waypoints are:

| Segment | Action chunk | Effective actions | Raw env steps | Spatial Δ |
|---|---|---|---|---|
| 0 → 1 | frames 0–3 | 4 | 20 | (-35, -15) |
| 1 → 2 | frames 4–10 | 7 | 35 | (-53, -9) |
| 2 → 3 | frames 11–15 | 5 | 25 | (-25, +27) |
| 3 → 4 | frames 16–18 | 3 | 15 | (-1, -1.3) |

The Stage-2 training task for this sample is:

```text
A_psi sees 4 different action chunks (lengths 4, 7, 5, 3 effective actions each)
  -> compresses each into one macro-action  l_0, l_1, l_2, l_3

E encodes the 5 waypoint images -> z_0, z_1, z_2, z_3, z_4

P^2 sees the pairs (z_0, l_0), (z_1, l_1), (z_2, l_2), (z_3, l_3)
  -> predicts  ẑ_1, ẑ_2, ẑ_3, ẑ_4

Loss:  L1( [ẑ_1, ẑ_2, ẑ_3, ẑ_4] , [z_1, z_2, z_3, z_4] )
```

Concretely: "given the agent is at (183, 74) and given a summary of the 20 raw env steps it took next, predict the encoded image of where it ended up". The macro-action `l_0` is just a learned 4-d vector — it's not "move left 35 pixels", it's whatever the network finds useful for predicting waypoint latents.

## 7. Why these numbers matter for TwoRoom specifically

| Fact | Why it matters |
|---|---|
| Mean episode = 92 raw steps; `span=100` keeps only 6,165/10,000 episodes | Stage-2 has **60× less data than Stage-1** (12K clips vs 731K). Plenty for 50 epochs but it's a real constraint. |
| TwoRoom episode is roughly "navigate from one position to another, maybe through a doorway" | A trajectory has very low intrinsic dimensionality — basically a 2D spatial path. The 4-dim macro-action has plenty of capacity to summarize ~25 raw env steps of movement. |
| Action range is `[-1, 1]` for each of 2 dims | The StandardScaler in `train_hierarchical.py` standardizes these to roughly zero-mean unit-variance; that's why CEM in the planner starts from `N(0, 0.1)` rather than from a wide prior. |
| Variable segment lengths (3–11 effective actions in the example above) | `A_psi` is a transformer with optional padding mask precisely for this. In the current code, padding is never actually needed because [waypoint_idx is sampled once per batch](../hierarchical_lewm.py:578) so every trajectory in the batch shares the same segment lengths. |
| Median episode = 101, max = 101 | The behavior policy / env has a hard cap around 101 steps. The data is roughly truncated/uniform around that length. |

## 8. One easy gotcha to remember

When you read `T = batch["pixels"].shape[1] = 20` inside `forward_high`, **that 20 is frames, not raw env steps**. Each frame represents 5 ticks of the simulator. So when `sample_waypoints` picks `[0, 4]` as a segment, "between waypoint 0 and waypoint 1" is **20 raw env steps**, not 4. That's why even with `n_waypoints=3` (only 5 total waypoints) and `stage2_num_steps=20`, each macro-action ends up summarising a meaningful chunk of physical motion.
