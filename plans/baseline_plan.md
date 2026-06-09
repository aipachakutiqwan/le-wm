# LeWM Baseline Generation Plan

> Goal: produce a reproducible, fully-trained **flat LeWM** baseline on Two-Room
> for the H-LeWM comparison described in [plan.md](plan.md). This document
> specifies the **experiment settings** for that baseline.
>
> Phase 0 (immediate) is a **1-epoch throughput benchmark** to size the full
> training run before committing GPU hours. Phase 1 (full training) and Phase 2
> (eval) will be parameter-locked once Phase 0 reports back wall-clock per
> epoch.

---

## 0. Scope

| Item | Decision |
|---|---|
| Environments | **Two-Room only** (the only `.h5` present locally; `tworoom.h5` at `$STABLEWM_HOME/tworoom.h5`) |
| Phase 0 epochs | **1** (benchmark / time-estimation) |
| Phase 1 epochs | TBD after Phase 0 (config default is 100; LeWM paper trains until convergence) |
| Seeds | **1 seed** (`seed=3072`, the config default) until we see borderline H-LeWM gains; revisit Step 14 of [plan.md](plan.md) for ≥3 seeds |
| Planner for eval | flat CEM-MPC (the existing `swm.solver.CEMSolver`) — H-LeWM hierarchical eval comes later |
| Logging | W&B → `florenciopaucar-uni/lewm` |
| Hardware | 1× CUDA GPU (already verified: torch 2.11+cu128, bf16) |

PushT / Reacher / OGB-Cube baselines are **out of scope** until their datasets
are downloaded.

---

## 1. Frozen experiment settings (do not change without revisiting this doc)

All values below come from `config/train/lewm.yaml` and `config/train/data/tworoom.yaml`. They match the LeWM paper. They are listed here so the baseline run is reproducible from this file alone.

### 1.1 Model

| Component | Setting |
|---|---|
| Encoder | ViT-tiny (`spt.backbone.utils.vit_hf("tiny", patch_size=14, image_size=224, pretrained=False)`); hidden = 192, 12 layers, 3 heads |
| Latent dim `D_z` | 192 (= ViT hidden = `embed_dim`) |
| Projector | `MLP(192 → 2048 → 192, BatchNorm1d)` |
| Predictor `P^(1)` | `ARPredictor(num_frames=3, input_dim=192, hidden_dim=192, output_dim=192, depth=6, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1)` with AdaLN-zero conditioning |
| Pred projector | `MLP(192 → 2048 → 192, BatchNorm1d)` |
| Action embedder `A_emb` | `Embedder(input_dim=10, emb_dim=192)` (Conv1d k=1 + 2-layer MLP) |
| History size | 3 |
| Num preds | 1 |
| Total trainable params | ~15 M |

### 1.2 Loss

| Term | Setting |
|---|---|
| `L_pred` | `MSE(pred_emb, tgt_emb).mean()` over `(B, T=3, D_z)` |
| SIGReg | `SIGReg(knots=17, num_proj=1024)` on `emb.transpose(0,1)` → `(T, B, D)` |
| Weight `λ` | **0.09** (config default; the paper quotes 0.1 — we follow the config) |
| Total | `L = L_pred + λ · L_sigreg` |

### 1.3 Optimizer & schedule

| Item | Value |
|---|---|
| Optimizer | `AdamW(lr=5e-5, weight_decay=1e-3)` |
| Scheduler | `LinearWarmupCosineAnnealingLR` (epoch interval; warmup + cosine over total epochs) |
| Gradient clipping | `gradient_clip_val=1.0` |
| Precision | `bf16` |
| Devices | `auto` (1 GPU here) |

### 1.4 Data

| Item | Value |
|---|---|
| Dataset | `tworoom` (HDF5; **730,809** transitions; 1 GPU env, no `state` col — only `proprio`) |
| Keys loaded | `pixels`, `action`, `proprio` |
| Keys cached in RAM | `action`, `proprio` |
| Image size | 224 × 224 (ImageNet-normalized) |
| `frameskip` | 5 (effective `D_a = frameskip · raw_action_dim = 5 · 2 = 10`) |
| `num_steps` per sample | 4 (= `history_size + num_preds`) |
| Train / val split | 0.9 / 0.1 via `spt.data.random_split(..., generator=torch.Generator().manual_seed(3072))` |
| Loader | `batch_size=128`, `num_workers=6`, `persistent_workers=True`, `prefetch_factor=3`, `pin_memory=True`, `shuffle=True`, `drop_last=True` |
| Action NaN handling | `torch.nan_to_num(action, 0.0)` inside `lejepa_forward` (last-step boundary action is NaN by convention) |

### 1.5 Logging & checkpointing

| Item | Setting |
|---|---|
| W&B entity / project | `florenciopaucar-uni` / `lewm` |
| W&B run name | `${output_model_name}` = `lewm` |
| W&B run id | `${subdir}` = `${hydra:job.id}` (Hydra-assigned per launch) |
| Per-step logs | `train/pred_loss`, `train/sigreg_loss`, `train/loss` (and `val/*` mirrored each epoch) |
| Checkpoint cadence | every epoch (`ModelObjectCallBack(epoch_interval=1)`) |
| Checkpoint location | `$STABLEWM_HOME/<hydra:job.id>/lewm_epoch_<k>_object.ckpt` + `lewm_weights.ckpt` |

---

## 2. Phase 0: 1-epoch throughput benchmark

**Purpose:** measure wall-clock per epoch (so Phase 1 epoch count is informed), verify W&B push, verify checkpoint serialization, verify CUDA bf16 stability on this box.

**Command** (run from repo root, in a shell that has the updated `STABLEWM_HOME`):

```powershell
C:\Users\kaboo\le-wm\.venv\Scripts\python.exe train.py `
    data=tworoom `
    trainer.max_epochs=1 `
    trainer.devices=1 `
    wandb.enabled=True
```

All other hyperparameters are taken from §1 (i.e. from `config/train/lewm.yaml`).

**What we measure during Phase 0**
- Wall-clock per epoch (target: ≤ 30 min on 1 GPU; if much higher, drop `loader.batch_size` or `loader.num_workers` and re-measure).
- GPU memory headroom (so we can grow batch size in Phase 1 if useful).
- `train/pred_loss` curve over the epoch — should monotonically decrease from random init (smoke tests already showed pred 0.13 → 0.008 in 20 SGD steps).
- `train/sigreg_loss` — should remain roughly stable (~3 on tworoom per the real-data smoke test); growth would suggest representation collapse risk.
- The `*_object.ckpt` file exists and is loadable via `torch.load(path, weights_only=False)`.

**Acceptance criteria for Phase 0**
- Training completes without exception.
- W&B run shows up at `https://wandb.ai/florenciopaucar-uni/lewm` with non-empty `train/loss` scalar.
- A `lewm_epoch_1_object.ckpt` exists under `$STABLEWM_HOME/<hydra:job.id>/`.
- Wall-clock-per-epoch number is recorded (in this file, see §4).

---

## 3. Phase 1: full training (parameters locked after Phase 0)

**To be filled in after Phase 0.** Decisions that depend on Phase 0's measured throughput:
- `trainer.max_epochs`: target a fully-trained baseline. Config default = 100; we'll keep that unless 1 epoch already takes hours.
- Whether to widen the batch (if GPU memory allows) for faster wall-clock convergence.
- Whether to add a second seed for variance — gated on remaining GPU budget.

The plan is **not** to change any hyperparameter listed in §1 between Phase 0 and Phase 1 — only the epoch budget and (possibly) batch size.

---

## 4. Phase 0 results (fill in after running)

```
hydra job id   :
wall-clock     :       (seconds for 1 epoch, training only)
final train/pred_loss   :
final train/sigreg_loss :
final train/loss        :
val/pred_loss           :
GPU peak memory  :       (MiB)
W&B run URL    :
checkpoint path  :       $STABLEWM_HOME/<job-id>/lewm_epoch_1_object.ckpt
notes / issues   :
```

---

## 5. Phase 2: baseline evaluation (after Phase 1 checkpoint exists)

**Eval settings** (from `config/eval/tworoom.yaml`, all values frozen):

| Item | Value |
|---|---|
| Env | `swm/TwoRoom-v1` |
| Parallel envs | `num_envs = num_eval = 50` |
| `max_episode_steps` | `2 · eval_budget` (set at runtime) |
| Planner | flat CEM-MPC (`swm.solver.CEMSolver`) |
| `plan_config.horizon` | 5 |
| `plan_config.receding_horizon` | 5 |
| `plan_config.action_block` | 5 (= `frameskip`) |
| `num_samples` (CEM) | 300 |
| `n_steps` (CEM iters) | 30 |
| `topk` (CEM elites) | 30 |
| `var_scale` | 1.0 |
| Eval episodes | 50 (random sample from valid starting points) |
| `goal_offset_steps` | 25 |
| `eval_budget` | 50 env steps |
| Seed | 42 (config default for eval) |
| State init | `_set_state(state=proprio)` (no `state` column in `tworoom.h5` — uses `proprio`) |
| Goal state | `_set_goal_state(goal_state=goal_proprio)` |

**Command**

```powershell
C:\Users\kaboo\le-wm\.venv\Scripts\python.exe eval.py `
    --config-name=tworoom.yaml `
    policy=<hydra-job-id>/lewm
```

**Outputs**
- `$STABLEWM_HOME/<hydra-job-id>/tworoom_results.txt` — appended with config + success rate + wall-clock.
- `rollout_*.mp4` videos in the same directory.

**This becomes the reference number we compare H-LeWM Stage 2 against.**

---

## 6. Pre-flight checklist (before kicking off Phase 0)

- [ ] `STABLEWM_HOME` in the launching shell resolves to `C:\Users\kaboo\.stable_worldmodel` (the current agent process has a stale value — start training in a fresh terminal, or prefix the command).
- [ ] `wandb login` has been run as `aj2622`, with credentials persisted to `C:\Users\kaboo\.netrc`.
- [ ] `tests/test_train_smoke.py` passes (synthetic + real). Already verified.
- [ ] Disk free under `$STABLEWM_HOME` is enough for one `*_object.ckpt` (≈ tens of MB for ~15 M params).

---

## 7. Known gotchas

- `tworoom.h5` does **not** have a `state` column; the eval config uses `proprio` for the env-state callable. PushT/Cube/Reacher configs reference `state` — when those datasets are added later, double-check column availability.
- `cfg.data.dataset.frameskip` × `cfg.wm.action_dim` must equal the dataset's stored action width (10 for tworoom). `train.py` sets `cfg.wm.<col>_dim` from `dataset.get_dim(col)` automatically, but the predictor is built before this side-effect runs, so action_dim must already be correct in the data yaml — it is (`wm.action_dim: 2` implied via tworoom yaml).
- The first epoch will rebuild the encoder normalizer cache; subsequent epochs are faster.
- `triton not found` warning is harmless (flop counting only); ignore.
