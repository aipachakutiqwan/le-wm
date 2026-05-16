# H-LeWM Implementation Plan

> Status: Stage 1 codebase mapping + Stage 2 implementation outline.
> This file is informational only. No source code has been modified.

---

## 0. Executive Summary

**What the existing codebase implements.**
The repository contains the minimal training/evaluation/planning glue for
**LeWorldModel (LeWM)**, a Joint-Embedding Predictive Architecture (JEPA)
trained end-to-end from pixels with two loss terms: a next-embedding MSE
prediction loss plus an isotropic-Gaussian latent regularizer (**SIGReg**).
The model architecture itself lives in three local files:
[jepa.py](jepa.py), [module.py](module.py), and the training/eval/utility
glue in [train.py](train.py), [eval.py](eval.py), [utils.py](utils.py).

Everything else (datasets, environments, the CEM-MPC planner, video
recording, evaluation harness, ViT backbone) is provided by two external
dependencies installed into `.venv/`:

- `stable_worldmodel` (planner + envs + datasets + policy wrappers).
- `stable_pretraining` (Lightning module wrapper + ViT helper + data transforms).

**Where the LeWM baseline lives.**

- LeWM model wiring and the `encode → predict → rollout → criterion → get_cost`
  surface: [jepa.py](jepa.py) (class `JEPA`).
- LeWM building blocks (SIGReg, predictor, action embedder, MLP heads):
  [module.py](module.py).
- LeWM training loop and loss: [train.py](train.py) function `lejepa_forward`.
- LeWM evaluation/planning: [eval.py](eval.py) + `swm.policy.WorldModelPolicy`
  + `swm.solver.CEMSolver` (flat CEM).

**What must remain unchanged for H-LeWM.**

- `JEPA.encode`, `JEPA.predict`, the ViT encoder, the `ARPredictor`
  architecture, the SIGReg loss, the existing `lejepa_forward` Stage-1 loss,
  and the `JEPA.get_cost` interface that the existing flat CEM solver depends on.
- The existing `config/train/lewm.yaml` and per-env eval configs should still
  produce the original baseline when `use_hierarchical_planning=false`.

**What will be added for H-LeWM.**

A second hierarchical level on top of the trained LeWM:

1. An **action encoder** `A_psi` that compresses chunks of primitive actions
   into latent macro-actions `l ∈ R^{D_l}`.
2. A **high-level predictor** `P^(2)` that predicts the next waypoint latent
   `z_{k+1}` from the current waypoint latent `z_k` and a macro-action `l_k`.
3. A **waypoint-subsampled** Stage-2 dataset/loader that yields
   `(waypoint observations, action chunks)`.
4. A **Stage-2 trainer** that freezes the encoder `E` and low-level predictor
   `P^(1)` and trains only `A_psi + P^(2)` with a teacher-forced L1 loss in
   latent space.
5. A **two-level CEM-MPC planner** (`plan_hierarchical`): outer CEM optimizes
   macro-action sequences; the first predicted waypoint becomes the subgoal
   for an inner CEM that reuses the existing flat CEM planner.

**Main implementation risk areas.**

- LeWM's latent space is shaped by SIGReg (approximately isotropic Gaussian).
  An L1 distance over `D_z=192` may be a poor proxy for environment success;
  needs early diagnostic.
- Variable-length action chunks between waypoints require careful padding
  + attention masks in `A_psi`.
- LeWM uses a 3-frame history (`history_size=3`); the high-level predictor
  must be redesigned to either consume waypoint embeddings directly (no
  history) or maintain its own short history.
- The flat CEM in `swm.solver.CEMSolver` is hard-coded to sample
  `(Batch, NumSamples, Horizon, action_block * action_dim)` from a normal
  distribution. Reusing it for "latent macro-action" CEM requires either
  subclassing it or feeding it through a fake `action_space`.
- Trajectory lengths in current datasets may be too short to demonstrate
  long-horizon hierarchical gains. Confirm before claiming results.

---

## 1. Paper Understanding: LeWorldModel

LeWorldModel (LeWM, arXiv: 2603.19312) is a JEPA that learns a latent
world model end-to-end from raw pixels. The model is a single `nn.Module`
that bundles:

- an **encoder** `E` (ViT) that maps an image `o` to a latent vector
  `z = E(o) ∈ R^{D_z}`;
- an **action embedder** `A_emb` that maps a primitive action `a ∈ R^{D_a}` to
  an action token `e_a ∈ R^{D_z}`;
- an **action-conditioned predictor** `P^(1)` that, given a short history of
  latents and action tokens, predicts the next latent `ẑ_{t+1}`.

The training objective has two terms:

- a next-embedding **MSE prediction loss** `||ẑ_{t+1} − z_{t+1}||²`;
- a **SIGReg** term that forces the marginal distribution of `z` over the
  batch+time dimension to look isotropic Gaussian (this prevents
  representation collapse without EMAs or auxiliary networks).

The full loss is `L = L_pred + λ · L_sigreg` (`λ = 0.09` in the default
config). This is "two terms / one hyperparameter" — the simplification
that the paper emphasizes versus PLDM/iJEPA/DINO-WM.

At evaluation time, LeWM is used as a cost model for **CEM-MPC**:
encode the current image and the goal image, roll out the predictor
autoregressively under each candidate action sequence, score sequences by
latent distance to the goal at the final step, and execute the first
chunk of the best sequence under receding horizon.

Long-horizon planning suffers because:
- the predictor is trained only on short subsequences (`history_size=3`,
  `num_preds=1`), so multi-step rollouts compound prediction error;
- the CEM search space grows linearly with horizon, so coverage drops fast.
HWM-style hierarchy is exactly the standard remedy.

### Mapping LeWM concepts to the local code

| Paper concept | Expected implementation role | Actual file path | Actual class/function | Status | Notes for someone re-learning PyTorch |
|---|---|---|---|---|---|
| JEPA-style latent prediction | Wraps E, P, A_emb, computes loss in latent space | [jepa.py](jepa.py) | `JEPA(nn.Module)` | FOUND | `nn.Module` is the base class for any PyTorch model. `JEPA.__init__` registers submodules as attributes; they are auto-discovered by `parameters()`, `to(device)`, etc. |
| Encoder `E` (image → latent) | Vision Transformer, returns CLS token | [train.py](train.py) (built), [jepa.py](jepa.py) (used) | `spt.backbone.utils.vit_hf(cfg.encoder_scale, ...)` instantiated in `train.py`; used in `JEPA.encode` as `self.encoder(pixels, interpolate_pos_encoding=True).last_hidden_state[:, 0]` | FOUND | Output is a HuggingFace `ModelOutput` with `last_hidden_state` of shape `(B*T, P+1, hidden)`. `[:, 0]` selects the CLS token. |
| Latent representation `z` | `D_z`-dim vector per frame | [jepa.py](jepa.py) | `info["emb"]` produced in `JEPA.encode`, shape `(B, T, D_z)` where `D_z = embed_dim = 192` | FOUND | `embed_dim` is set in `config/train/lewm.yaml` (`wm.embed_dim: 192`); equals ViT-tiny hidden size. |
| Latent projection head | Optional linear/MLP head after CLS token | [jepa.py](jepa.py), [module.py](module.py) | `JEPA.projector` (an `MLP` with BatchNorm1d) + `JEPA.pred_proj` (same architecture, applied after predictor) | FOUND | Both are `MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=BatchNorm1d)`. If `hidden_dim == embed_dim` they could become `nn.Identity()`. |
| Action-conditioned predictor `P^(1)` | Autoregressive transformer with AdaLN-zero on action | [module.py](module.py) | `ARPredictor` | FOUND | Uses `ConditionalBlock` with AdaLN-zero modulation (`adaLN_modulation` projects condition `c` to `6*dim` for shift/scale/gate on attn+MLP). |
| Action embedder `A_emb` | Maps primitive action to action token | [module.py](module.py) | `Embedder` (Conv1d patch embed + 2-layer MLP) | FOUND | Important: `Embedder` is reused across the time axis. The effective action dim is `frameskip * action_dim` because `dataset.frameskip=5` reshapes a chunk of raw actions into a single grouped action. |
| Prediction loss `L_pred` | MSE between predicted latent and target latent | [train.py](train.py) | `lejepa_forward`: `output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()` | FOUND | `pred_emb` shape `(B, history_size, D_z)`, `tgt_emb` is `emb[:, num_preds:]` i.e. shifted by 1 along time. |
| SIGReg | Sketch isotropic Gaussian regularizer | [module.py](module.py) | `SIGReg` | FOUND | Computes the Epps-Pulley statistic on `num_proj=1024` random unit projections of the latents, using `knots=17` integration points. Inputs are `(T, B, D)` (note the transpose in `train.py`: `self.sigreg(emb.transpose(0, 1))`). It uses `register_buffer` for non-learnable tensors (knot positions, weights). |
| Two-term objective | Combined loss | [train.py](train.py) | `output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]` with `lambd = cfg.loss.sigreg.weight = 0.09` | FOUND | This is the scalar that `pl.Trainer` calls `.backward()` on. Lightning handles `optimizer.zero_grad / step` automatically. |
| CEM-MPC planning loop | Outer MPC + iterative CEM inside | external (`stable_worldmodel`) | `swm.solver.CEMSolver` + `swm.policy.WorldModelPolicy` | FOUND | `CEMSolver.solve` samples `(B, num_samples, horizon, action_dim)` candidates, calls `model.get_cost(info, candidates)`, updates `mean`/`var` from top-K. `WorldModelPolicy.get_action` calls `solve` and pops `receding_horizon * action_block` actions from a buffer. |
| Goal image encoding | Encode `goal` image through the same encoder | [jepa.py](jepa.py) | `JEPA.get_cost` constructs a `goal` dict from `info_dict["goal"]` and calls `self.encode(goal)` — see lines that copy `goal_*` keys to `*` and `goal["pixels"] = goal["goal"]`. | FOUND | The goal is re-encoded each call. Caching is not implemented in this local `JEPA` (`PreJEPA` in `swm.wm.prejepa` does cache via `_goal_cached_info`). |
| Planning cost | Latent MSE between final predicted emb and goal emb | [jepa.py](jepa.py) | `JEPA.criterion`: MSE on last time-step latent only | FOUND | `cost` shape `(B, S)`. The reduction sums over the feature dim and squeezes out the time dim (last-step only). |
| Compounding prediction error | Autoregressive rollout `H` steps with shared `P^(1)` | [jepa.py](jepa.py) | `JEPA.rollout` (loop over `n_steps`) | FOUND | History truncation: each step uses only the last `history_size=3` latents + actions. Errors accumulate because predicted latents are fed back as inputs. |
| Optimizer construction | AdamW + LinearWarmupCosineAnnealingLR | [train.py](train.py) + `stable_pretraining` | `optimizers = {"model_opt": {...}}` dict passed to `spt.Module(... optim=optimizers)` | FOUND | This is `stable_pretraining`'s declarative optimizer spec; the actual `torch.optim.AdamW` is constructed inside `spt.Module`. |
| Checkpoint saving | Pickle the world-model object every epoch | [utils.py](utils.py) | `ModelObjectCallBack` (a Lightning `Callback`) | FOUND | Calls `torch.save(model, path)` — saves the full module *object* (not just `state_dict`). Required by `swm.policy.AutoCostModel`, which loads the object and looks for any submodule with a `get_cost` method. |
| Evaluation mode / `torch.no_grad` | Solver runs under `@torch.inference_mode()` | external | `CEMSolver.solve` is decorated with `@torch.inference_mode()`; `eval.py` puts model in `.eval()` and calls `model.requires_grad_(False)` | FOUND | `torch.inference_mode()` is strictly stronger than `torch.no_grad()` and is the recommended decorator for pure-inference paths in modern PyTorch. |

---

## 2. Paper Understanding: HWM

Hierarchical Planning with Latent World Models (HWM, arXiv 2604.03208)
adds **multiple temporal abstraction levels** on top of a pretrained
latent world model. Concretely, HWM keeps the original low-level
predictor `P^(1)` (one primitive-action step per call) and adds:

- a **high-level predictor** `P^(2)` that operates between
  *waypoints* spaced `T_chunk` primitive steps apart;
- an **action encoder** `A_psi` that maps the chunk of `T_chunk` primitive
  actions between two consecutive waypoints to a single **latent
  macro-action** `l ∈ R^{D_l}`.

`P^(2)(z_k, l_k) → ẑ_{k+1}` is trained with **teacher forcing**: the
ground-truth waypoint latents come from re-encoding the waypoint
observations with the frozen encoder, the macro-actions come from
encoding the ground-truth primitive-action chunks with `A_psi`, and the
loss is L1 between `ẑ_{k+1}` and `E(o_{k+1})`. The encoder `E` and the
low-level predictor `P^(1)` are frozen during Stage 2.

At planning time, HWM uses a **two-level CEM**:

1. **Outer CEM** samples macro-action sequences `l_{1..H_high}` from a
   diagonal Gaussian in `R^{H_high × D_l}`, rolls them out through
   `P^(2)` from the current encoded latent, and scores each candidate by
   latent distance to the encoded goal at horizon `H_high`. The best
   macro-action sequence's *first* predicted latent `ẑ_1` is interpreted
   as a **subgoal** for the next replan.
2. **Inner CEM** samples primitive-action sequences of length `H_low`,
   rolls them out through `P^(1)`, and scores them by latent distance to
   `ẑ_1`. Only the first primitive action is executed; then everything
   replans.

Hierarchy reduces the effective search dimensionality from
`H_total × D_a` (flat) to roughly `H_high × D_l + H_low × D_a` where
typically `H_high · T_chunk ≈ H_total` but `D_l ≪ T_chunk · D_a` and
`H_low` is short.

HWM's experimental validation:
- Tested on **multiple JEPA-family world models**: VJEPA2-AC, DINO-WM, and
  **PLDM**. PLDM is the closest architectural analogue to LeWM (small
  end-to-end JEPA trained from pixels with a simple loss). It is **not**
  the case that HWM only works with frozen pretrained encoders — what is
  required is that the low-level model be already trained and stable.
- The novelty for *this* project is therefore: does HWM-style
  hierarchical latent planning still work when the low-level model is
  LeWM (which uses SIGReg, end-to-end from pixels, ~15M params)?

### Mapping HWM concepts to LeWM

| Paper concept | How it applies to LeWM | Expected code location / integration point | Already supported? | What needs to be added | Notes for someone re-learning PyTorch |
|---|---|---|---|---|---|
| Multiple temporal abstraction levels | Keep LeWM's `P^(1)` as the low level; add `P^(2)` at waypoint spacing `T_chunk` | New module `hlewm/high_level.py` (proposed); wires in `train.py` and a new `train_stage2.py` | NO | New `nn.Module` for `P^(2)`; can reuse `ARPredictor` from [module.py](module.py) with `input_dim = embed_dim`, `action_dim = latent_action_dim` | Reuse via composition, not inheritance: instantiate a second `ARPredictor` with different `cond_proj`. |
| Low-level predictor `P^(1)` | Exactly LeWM's existing `ARPredictor` + `pred_proj`, frozen in Stage 2 | [jepa.py](jepa.py) `JEPA.predictor`, `JEPA.pred_proj`; [module.py](module.py) `ARPredictor` | YES (predictor exists) | `requires_grad_(False)` + `.eval()` switch around it during Stage 2 | Freezing alone doesn't disable BatchNorm running stats — must also call `.eval()` on the projector/pred_proj (which use BatchNorm1d). |
| High-level predictor `P^(2)` | Same `ARPredictor` family, but consumes waypoint-spaced latents and `D_l`-dim macro-action tokens | New: `hlewm.modules.HighLevelPredictor` (wraps `ARPredictor`) | NO | Class + a thin wrapper that calls `self.predictor(z_k, l_k)` autoregressively | Need to pick whether `P^(2)` keeps a short history (HWM's PLDM variant) or is single-state Markov. Default plan: single-state Markov, history_size=1. |
| Latent macro-action `l` | `D_l`-dim vector summarizing `T_chunk` primitive actions | Output of `A_psi`; input to `P^(2)`'s `c` argument | NO | New module; new config field `latent_action_dim` (`D_l`, default 8) | Don't conflate with `Embedder` (action_embedder in LeWM): that one embeds a *single* (possibly frameskip-grouped) action. `A_psi` compresses a *sequence* of those. |
| Action encoder `A_psi` | Small Transformer encoder pooled via `[CLS]` token over action chunk | New: `hlewm.modules.ActionEncoder` | NO | Class. Input `(B, N, L_chunk, D_a)`, output `(B, N, D_l)` | Use a padding mask: chunks between waypoints can vary in length. Mask flows into `F.scaled_dot_product_attention(key_padding_mask=...)`. |
| Waypoint-subsampled training | Pick `N` waypoints in a long trajectory and form `(z_k, action_chunk_k, z_{k+1})` triples | New: `hlewm.data.WaypointSampler` + `hlewm.data.Stage2Collator` | NO | Sampler + collator on top of `swm.data.HDF5Dataset` (long-episode mode) | The existing `swm.data.HDF5Dataset` returns short slices of `num_steps = history_size + num_preds = 4`. Stage 2 needs full episodes or much longer slices — likely override `num_steps` and `frameskip` or use `dataset._load_slice` directly. |
| Teacher-forcing loss for `P^(2)` | L1 between `P^(2)(E(o_k), A_psi(a_{k:k+T_chunk})) ` and `E(o_{k+1})` | New: `hlewm.losses.high_level_loss` (and Stage-2 forward in `train_stage2.py`) | NO | A `forward` fn analogous to `lejepa_forward` | Use `F.l1_loss` (default `reduction='mean'`); detach `E(o_k)` and `E(o_{k+1})` because the encoder is frozen — but detaching is belt-and-suspenders since `requires_grad_(False)` already blocks gradient flow. |
| Two-level CEM planning | Outer CEM over macro-actions; inner CEM over primitives | New: `hlewm.planner.HierarchicalPlanner` reusing `swm.solver.CEMSolver` twice (or a custom wrapper) | PARTIAL (flat CEM exists) | New `Costable`-compatible wrappers for both levels; new top-level `get_cost`/`get_action` flow | The existing `CEMSolver` assumes `Box` action space with shape `(action_block, action_dim)`. For the outer CEM the action space is `R^{D_l}` per step with `action_block=1`. Either fake the `gym.Space` or subclass `CEMSolver`. |
| Outer CEM over macro-actions | Sample `(S, H_high, D_l)` candidate macro-action sequences from `N(0, I)`; roll out `P^(2)`; cost to goal | New: `hlewm.planner.OuterCEM` | PARTIAL | Mostly reuse `CEMSolver` with a "fake" action space `Box(low=-inf, high=inf, shape=(1, D_l))` and a custom `Costable` adapter | Macro-actions are latent — no need to renormalize via `process['action']`; planner should *not* be wrapped in `process['action'].inverse_transform`. |
| Inner CEM over primitive actions | Reuse existing flat CEM solver with the subgoal latent as the planning goal | [eval.py](eval.py) + `swm.solver.CEMSolver` | PARTIAL | A `Costable` that overrides `criterion` so that the "goal latent" is `ẑ_subgoal` (returned by outer CEM) rather than the encoded goal image | Just need to skip the `goal_emb = encode(goal_image)` step in `JEPA.get_cost` and inject `ẑ_subgoal` directly. |
| Hierarchy reduces search dim | Outer search uses `H_high·D_l` dims, inner uses `H_low·D_a` dims | conceptual | YES | Verify numerically: `H_high=3, D_l=8` vs flat `H_total=15, D_a=2` for PushT | Useful to log this metric in W&B. |
| First high-level latent → low-level subgoal | After outer CEM, use predicted `ẑ_1` (NOT `ẑ_{H_high}`) as the inner CEM's goal latent | `hlewm.planner.plan_hierarchical` | NO | New code | Receding horizon at the high level: only the first waypoint matters because we will replan after a short execution. |

---

## 3. Repository Map

### 3.1 High-level directory tree

```text
le-wm/
├── README.md                ← LeWM paper abstract, usage, ckpt links
├── CONTRIBUTING.md          ← dev workflow (uv, pre-commit, commitizen)
├── LICENSE
├── pyproject.toml           ← deps (hydra, lightning, stable_*, torch>=2.11)
├── uv.lock
├── jepa.py                  ← class JEPA(nn.Module): encode/predict/rollout/criterion/get_cost
├── module.py                ← SIGReg, ARPredictor, Embedder, MLP, Transformer blocks
├── train.py                 ← Hydra @main; builds dataset+model+optim, calls spt.Manager
├── eval.py                  ← Hydra @main; builds World+Policy+CEMSolver, calls evaluate_from_dataset
├── utils.py                 ← img preprocessor, column normalizer, ModelObjectCallBack
├── config/
│   ├── train/
│   │   ├── lewm.yaml        ← top-level training cfg
│   │   ├── data/{pusht,dmc,ogb,tworoom}.yaml
│   │   └── launcher/local.yaml
│   └── eval/
│       ├── {pusht,cube,tworoom,reacher}.yaml
│       ├── solver/{cem,adam}.yaml
│       └── launcher/local.yaml
├── plans/
│   ├── .gitkeep
│   └── plan.md              ← THIS FILE
└── assets/lewm.gif
```

There is **no `tests/` directory** and **no Python tests** anywhere in the
repo root. Search confirmed (`*.py` `def test_` → zero matches outside
`.venv/`).

### 3.2 Entry points

| Purpose | Command | Path | Calls next | Important args | Output artifacts |
|---|---|---|---|---|---|
| Training | `python train.py data=pusht` | [train.py](train.py) | builds `swm.data.HDF5Dataset` → splits → `JEPA(...)` → `spt.Manager(trainer, module, data)()` | `data={pusht,dmc,ogb,tworoom}`, `trainer.max_epochs`, `loader.batch_size`, `loss.sigreg.weight`, `wandb.enabled` | `$STABLEWM_HOME/<hydra:job.id>/{lewm_epoch_*_object.ckpt, lewm_weights.ckpt, config.yaml}` |
| Evaluation/Planning | `python eval.py --config-name=pusht.yaml policy=pusht/lewm` | [eval.py](eval.py) | `swm.World(...)` + `swm.policy.AutoCostModel(cfg.policy)` + `hydra.utils.instantiate(cfg.solver, model=model)` + `swm.policy.WorldModelPolicy(...)` → `world.evaluate_from_dataset(...)` | `policy=<run_name>`, `eval.num_eval`, `eval.goal_offset_steps`, `eval.eval_budget`, `plan_config.{horizon,receding_horizon,action_block}` | `<results_path>/{pusht_results.txt, rollout_*.mp4}` |
| Dataset loading | (library) | `swm.data.HDF5Dataset` (in `.venv/Lib/site-packages/stable_worldmodel/data/dataset.py`) | `h5py.File` on `$STABLEWM_HOME/<name>.h5` | `name`, `frameskip`, `num_steps`, `keys_to_load`, `keys_to_cache` | – |
| Dataset generation | (library) | `swm.World.record_dataset(...)` | rollout policy and dump HDF5 | – | `<name>.h5` |
| Tests | none currently | – | – | – | – |

### 3.3 Model components

| Component | File | Class/Function | Input tensors | Output tensors | Important shapes | PyTorch concepts used | Paper concept | Notes |
|---|---|---|---|---|---|---|---|---|
| Image encoder | external | `spt.backbone.utils.vit_hf(scale='tiny', patch_size=14, image_size=224)` (called in [train.py](train.py) line ~82) | `pixels (B*T, 3, 224, 224)` (float) | HF `ModelOutput`; `.last_hidden_state` is `(B*T, P+1, hidden=192)` | hidden=192, patch_size=14 → 256 patches | HF ViT, `interpolate_pos_encoding=True` lets the model handle non-train image sizes | Encoder `E` | CLS token is index 0 (`output.last_hidden_state[:, 0]`). |
| Latent projector | [module.py](module.py) | `MLP(input_dim=hidden_dim, hidden_dim=2048, output_dim=embed_dim, norm_fn=BatchNorm1d)` instantiated in [train.py](train.py) as `projector=MLP(...)` | `(B*T, hidden=192)` | `(B*T, D_z=192)` | – | `nn.Sequential`, `BatchNorm1d` on flattened (B*T) dim | Latent projection head | Set to `nn.Identity` if `hidden==embed_dim` is desired; current default keeps both 192 so the MLP is essentially a learned re-mapping. |
| Predictor proj | [module.py](module.py) | same `MLP` class with the same shape; instantiated as `predictor_proj` in `train.py` | `(B*T, hidden)` | `(B*T, embed_dim)` | – | – | Output head of `P^(1)` | Applied in `JEPA.predict` after rearranging to `(b t) d`. |
| Predictor `P^(1)` | [module.py](module.py) | `ARPredictor(num_frames=3, input_dim=192, hidden_dim=192, output_dim=192, depth=6, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1)` | `x (B, T, D_z)`, `c (B, T, D_z)` | `(B, T, D_z)` | T=history_size=3 | `nn.Parameter` (`pos_embedding`), `Transformer` with `ConditionalBlock` (AdaLN-zero), `F.scaled_dot_product_attention(..., is_causal=True)` | `P^(1)` (action-conditioned) | Action conditioning is via AdaLN-zero modulation: `c` is projected to `6*dim` and used to shift/scale/gate attn+MLP. |
| Action embedder `A_emb` | [module.py](module.py) | `Embedder(input_dim=effective_act_dim, emb_dim=192)` instantiated as `action_encoder` in `train.py` | `(B, T, effective_act_dim=frameskip*action_dim)` | `(B, T, D_z=192)` | frameskip=5; e.g. PushT action_dim=2 → effective_act_dim=10 | `nn.Conv1d(kernel_size=1)` on `(B, D, T)`, then 2-layer MLP | Action embedding | Stored on the JEPA module as `self.action_encoder`. Re-used at both training and rollout time. |
| Loss: pred | [train.py](train.py) | `(pred_emb - tgt_emb).pow(2).mean()` inside `lejepa_forward` | `pred_emb (B, history_size, D_z)`, `tgt_emb (B, history_size, D_z)` | scalar | – | tensor arithmetic with autograd | Prediction MSE | Note `mean()` includes all dims — this normalizes by `B*T*D`, not by `B`. |
| Loss: SIGReg | [module.py](module.py) | `SIGReg(knots=17, num_proj=1024)` | `proj (T, B, D)` | scalar | – | `register_buffer` (knots, weights, phi), random unit-vector sampling, Epps-Pulley statistic | SIGReg | Buffers move with `.to(device)` but are not in `parameters()`. The random projections `A` are *fresh per call* — not learnable. |
| Total loss | [train.py](train.py) | `output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]` | scalars | scalar | – | autograd | Two-term objective | Lightning's `spt.Module` calls `.backward()` on this. |
| Optimizer | [train.py](train.py) | `{"type":"AdamW","lr":5e-5,"weight_decay":1e-3}` + `LinearWarmupCosineAnnealingLR` | – | – | – | declarative config in `spt.Module(... optim=...)` | – | Constructed inside `spt.Module`; you don't see `optim.AdamW(model.parameters(), ...)` explicitly here. |
| Checkpointing | [utils.py](utils.py) | `ModelObjectCallBack(dirpath, filename, epoch_interval=1)` | – | `<filename>_epoch_<k>_object.ckpt` | – | Lightning `Callback`; `torch.save(model, path)` saves the whole object | – | Object checkpoint is what `swm.policy.AutoCostModel` consumes. |
| Eval mode | [eval.py](eval.py) | `model.eval(); model.requires_grad_(False); model.interpolate_pos_encoding = True` | – | – | – | `.eval()` (toggles BN/Dropout), `requires_grad_(False)` (no grad tracking), `@torch.inference_mode()` on `CEMSolver.solve` | – | The `interpolate_pos_encoding = True` flag is a *Python attribute* assignment on the loaded module — not a config — and is used inside `JEPA.encode` via `self.encoder(pixels, interpolate_pos_encoding=True)`. |
| Device handling | [eval.py](eval.py) + [jepa.py](jepa.py) | `model = model.to("cuda")`; `JEPA.get_cost` moves tensors via `device = next(self.parameters()).device` | – | – | – | The idiom `next(self.parameters()).device` is the standard way to get a model's current device. | – | Trainer config sets `accelerator: gpu, precision: bf16`. |
| Batch handling | [train.py](train.py) | `torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True)` | – | – | batch_size=128, num_workers=6, prefetch_factor=3, pin_memory=True | DataLoader, dataset transforms via `spt.data.transforms.Compose` | – | `cfg.loader.persistent_workers=True` keeps workers alive across epochs. |

### 3.4 Training flow

Walk-through of [train.py](train.py) for a rusty PyTorch reader:

1. **Hydra entry** at `@hydra.main(...config_name="lewm")`. The `cfg`
   object is a nested `OmegaConf` DictConfig built by merging
   `config/train/lewm.yaml` and `config/train/data/<data>.yaml`. Default
   is `data=pusht`.
2. **Dataset**: `swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)`
   loads (or memory-maps) the `.h5` file at
   `$STABLEWM_HOME/<name>.h5`. With `num_steps=4` (=`history_size +
   num_preds`) and `frameskip=5`, each `__getitem__` returns a dict with
   tensors of shape `(num_steps, ...)`.
3. **Transforms**: An image preprocessor (`utils.get_img_preprocessor`)
   normalizes pixels with ImageNet stats and resizes to
   `cfg.img_size=224`. Additional columns (e.g. `proprio`, `state`,
   `action`) get a learned `StandardScaler`-style normalizer that fits on
   the cached column data and divides by std.
4. **Train/val split**: `spt.data.random_split(dataset, [0.9, 0.1],
   generator=...)` — uses a seeded `torch.Generator` so the split is
   reproducible.
5. **DataLoaders**: `torch.utils.data.DataLoader(...)`. The batch yielded
   by `train` is a dict whose tensors have shape `(B=128, T=4, ...)`. For
   PushT the `pixels` tensor will be `(B, T, 3, 224, 224)`, `action`
   `(B, T, effective_act_dim=10)`, etc.
6. **Model build**:
   - encoder = ViT-tiny (`spt.backbone.utils.vit_hf("tiny", patch_size=14,
     image_size=224, pretrained=False)`),
   - predictor = `ARPredictor(num_frames=3, input_dim=192, hidden_dim=192,
     depth=6, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1)`,
   - action_encoder = `Embedder(input_dim=10, emb_dim=192)`,
   - projector = `MLP(192 → 2048 → 192, BN)`,
   - pred_proj = same MLP class.
   These are bundled in `world_model = JEPA(...)`.
7. **Lightning wrapper**: `world_model = spt.Module(model=world_model,
   sigreg=SIGReg(...), forward=partial(lejepa_forward, cfg=cfg),
   optim=optimizers)`. `spt.Module` is a `pl.LightningModule`; it
   automatically registers `forward` as both `training_step` and
   `validation_step` (the `stage` argument distinguishes them).
8. **Forward step** (`lejepa_forward(self, batch, stage, cfg)`):
   - `batch["action"] = torch.nan_to_num(batch["action"], 0.0)` — last
     step's action is NaN by dataset convention; replace with 0.
   - `output = self.model.encode(batch)` produces `output["emb"]`
     `(B, T=4, D_z=192)` and `output["act_emb"]` `(B, T=4, D_z=192)`.
   - `ctx_emb = emb[:, :history_size=3]`, `ctx_act = act_emb[:, :3]`.
     `tgt_emb = emb[:, num_preds=1:]` = `emb[:, 1:]` = shape `(B, 3, D_z)`.
   - `pred_emb = self.model.predict(ctx_emb, ctx_act)` → `(B, 3, D_z)`.
   - `pred_loss = (pred_emb - tgt_emb).pow(2).mean()` — MSE.
   - `sigreg_loss = self.sigreg(emb.transpose(0, 1))` — note the
     `(T, B, D)` ordering required by `SIGReg.forward`.
   - `loss = pred_loss + lambd * sigreg_loss`.
   - Lightning takes `output["loss"]` and runs `.backward()` + optimizer
     step automatically.
9. **Logging**: `self.log_dict(losses_dict, on_step=True, sync_dist=True)`
   logs `train/pred_loss`, `train/sigreg_loss`, `train/loss` to W&B.
10. **Checkpointing**: `ModelObjectCallBack` (in [utils.py](utils.py))
    runs at the end of every train epoch and `torch.save(model, path)`s
    the full `spt.Module` object. The Manager additionally saves a
    weights-only checkpoint `<output_model_name>_weights.ckpt`.

### 3.5 Planning / CEM flow

Walk-through of [eval.py](eval.py) + the external solver:

1. **Hydra entry** with `config_name=pusht.yaml` (default; override with
   `--config-name`).
2. **World**: `swm.World(env_name='swm/PushT-v1', num_envs=cfg.eval.num_eval=50,
   max_episode_steps=2*eval_budget, image_shape=(224,224))`. World wraps
   a `SyncWorld` of `num_envs=50` parallel envs.
3. **Model load**: `model = swm.policy.AutoCostModel(cfg.policy)` walks
   the saved Lightning object and returns the first descendant with a
   `get_cost` method — that is exactly the `JEPA` instance from
   [jepa.py](jepa.py). It then `.to("cuda").eval()` and disables grads.
4. **Plan config**: `swm.PlanConfig(horizon=5, receding_horizon=5,
   action_block=5, warm_start=True)`. `plan_len = horizon *
   action_block = 25` env steps per plan.
5. **Solver**: `hydra.utils.instantiate(cfg.solver, model=model)` builds
   `CEMSolver(model=model, batch_size=1, num_samples=300, var_scale=1.0,
   n_steps=30, topk=30, device='cuda', seed=42)`.
6. **Policy wrap**: `policy = swm.policy.WorldModelPolicy(solver=solver,
   config=config, process=process, transform=transform)`. `process`
   contains learned per-column normalizers (StandardScaler) used to
   undo/redo normalization on actions and other tensors.
7. **Reset-and-step loop** (inside
   `world.evaluate_from_dataset`):
   - For each env step, `world.step()` calls
     `self.policy.get_action(self.infos)`.
   - `WorldModelPolicy.get_action`:
     - prepares info (image transforms + column scalers);
     - if the internal action buffer is empty, calls `solver.solve(info,
       init_action=self._next_init)`;
     - chops the returned plan into a buffer of `receding_horizon *
       action_block = 25` primitive actions;
     - pops one primitive action; `process['action'].inverse_transform`
       un-normalizes it before passing to `env.step`.
8. **CEM inner loop** (`swm.solver.CEMSolver.solve`):
   - Initialize `mean[B, H, D_a*action_block]`, `var[B, H, D_a*action_block]`.
   - For step in `range(n_steps=30)`:
     - Sample `candidates ~ N(mean, var)` of shape `(B, num_samples=300, H, D_eff)`.
     - Force `candidates[:, 0] = mean` (elite re-injection).
     - `costs = model.get_cost(expanded_infos, candidates)` (shape
       `(B, num_samples)`).
     - Pick top-K (lowest cost), recompute mean and std.
   - Return `outputs['actions'] = mean` of shape `(B, H, D_eff)`.
9. **Cost computation** (`JEPA.get_cost`):
   - Move all tensors in `info_dict` to model's device.
   - Build a `goal` sub-dict from `info_dict["goal"]` (and `goal_*`
     keys), encode it with `JEPA.encode` → `goal_emb (B, S, D_z)`.
   - Call `JEPA.rollout(info_dict, action_candidates, history_size=3)`:
     - encode the initial pixels (single frame) → repeat across S samples;
     - autoregressively roll the predictor for `n_steps = T_seq -
       history_size` steps; truncate history to last 3 latents at each step;
     - return `predicted_emb (B, S, ..., D_z)`.
   - `JEPA.criterion`: last-step MSE against `goal_emb[:, :, -1:, :]`,
     summed over feature dim → cost `(B, S)`.
10. **Replanning** is governed by `WorldModelPolicy._action_buffer`:
    once the buffer is empty (after 25 env steps), the policy calls
    `solver.solve` again with `init_action=_next_init` (warm start by
    shifting the previous plan).

**The planner is vectorized over samples (S) but iterates batch-of-1
across envs**: `batch_size=1` means `CEMSolver.solve` loops over the
50 envs sequentially. This is a key compute bottleneck and a place where
H-LeWM gains/losses will be measured.

### 3.6 Dataset and environment flow

| Aspect | Where | Details |
|---|---|---|
| Dataset class | `swm.data.HDF5Dataset` (external) | Single `.h5` file per dataset under `$STABLEWM_HOME` |
| Trajectory format | HDF5 `ep_offset`, `ep_len` indexes + per-column flat arrays | `_load_slice(ep_idx, start, end)` returns dict of `(num_steps, ...)` torch tensors |
| Observation format | `pixels: (T, 3, H, W) uint8` after permute (`HDF5Dataset._load_slice` permutes `(N,H,W,3)→(N,3,H,W)` when channels-last is detected) | resized to `224×224` and ImageNet-normalized by the transform pipeline |
| Action format | `action: (T*frameskip, action_dim)` flat in HDF5; reshaped by `Dataset.__getitem__` to `(num_steps, frameskip*action_dim)` | `frameskip=5` is the default in all train configs |
| Train/eval split | [train.py](train.py) | `spt.data.random_split(dataset, [0.9, 0.1], generator=Generator(seed))` |
| Supported environments | `config/eval/{pusht,cube,tworoom,reacher}.yaml` | `swm/PushT-v1`, `swm/OGBCube-v0`, `swm/TwoRoom-v1`, `swm/ReacherDMControl-v0` |
| Sequence length / horizon | training | `num_steps = history_size + num_preds = 4` |
| Sequence length / horizon | eval | `plan_config.horizon=5`, `action_block=5`, `eval.eval_budget=50` env steps |
| Push-T | YES | `config/train/data/pusht.yaml` + `config/eval/pusht.yaml` |
| OGBench-Cube | YES | `config/train/data/ogb.yaml` (ogbench/cube_single_expert) + `config/eval/cube.yaml` |
| Two-Room | YES | `config/train/data/tworoom.yaml` + `config/eval/tworoom.yaml` |
| Reacher | YES | `config/train/data/dmc.yaml` (reacher) + `config/eval/reacher.yaml` |
| Violation-of-Expectation (VoE) | NOT FOUND in this repo | LeWM paper mentions surprise evaluation; not implemented locally. |

### 3.7 Existing tests

There are **no tests** in this repository. Grep for `def test_` returns
zero matches outside `.venv/`. The `pyproject.toml` declares pytest as
an optional dev dep but `pytest -q` would currently collect nothing.

Implication: every H-LeWM step below must ship its own unit test in a
new `tests/` package (proposed: `tests/hlewm/test_*.py`).

---

## 4. Annotated Codebase Map for H-LeWM

This is the index of code locations that H-LeWM will touch. Nothing is
edited here; this is purely the map.

| Label | File | Class/Function | Current Behavior | Paper Concept | Why This Location Matters | Expected H-LeWM Change | Risk | How To Test |
|---|---|---|---|---|---|---|---|---|
| HLEWM-ENCODER | [jepa.py](jepa.py) + [train.py](train.py) | `JEPA.encode`, `JEPA.encoder` (a `vit_hf` instance) | Encodes pixels via ViT-tiny + CLS-token + `projector` MLP, into `(B,T,D_z=192)` | LeWM encoder `E` | Stage-2 needs the *same* encoder, frozen, to provide waypoint latents and goal latent for both inner and outer CEM | None to the module itself. Add a thin `encode_obs(obs)` helper that takes a single image tensor and returns `(B, D_z)` | low | Test: encode a fixed image twice and assert identical output (eval mode); assert grad is None when frozen |
| HLEWM-LOW-PREDICTOR | [jepa.py](jepa.py) + [module.py](module.py) | `JEPA.predict`, `JEPA.predictor` (`ARPredictor`), `JEPA.pred_proj` | One-step latent prediction conditioned on a 3-step history and action tokens | `P^(1)` | Inner CEM rolls out exactly this | None — wrap it in a clean `rollout_primitive(z_history, actions)` adapter that doesn't depend on `info_dict` | low | Test: random init `JEPA`, rollout 1 step, assert output shape `(B, D_z)` |
| HLEWM-SIGREG | [module.py](module.py) | `SIGReg(knots=17, num_proj=1024)` | Computes Epps-Pulley statistic on random unit projections | SIGReg loss | Stage-2 should *not* re-apply SIGReg to the high-level latents (they live in the same space, already regularized by Stage-1) | None | none if untouched; medium if accidentally double-applied | Test: assert `SIGReg(z)` is a scalar `Tensor` for `z` of shape `(T,B,D)` |
| HLEWM-LOW-LOSS | [train.py](train.py) | `lejepa_forward` | Stage-1 forward (pred_loss + sigreg_loss) | LeWM total loss | Must keep working for Stage-1 + baseline | None | none | Smoke test: load tiny batch, call `lejepa_forward`, assert loss is finite |
| HLEWM-CEM | external `swm.solver.cem.CEMSolver` | `CEMSolver.solve` | Iterative CEM with elite re-injection; assumes `model.get_cost(info, candidates) → (B,S)` | Flat CEM-MPC | Inner CEM will reuse it as-is; outer CEM will subclass or wrap | None (reuse) | medium: tight coupling to `Box` action space + `process['action']` post-processing | Test: call `CEMSolver.solve` on a toy quadratic cost, check convergence |
| HLEWM-MPC | external `swm.policy.WorldModelPolicy` | `WorldModelPolicy.get_action` | Action buffer + receding horizon | MPC outer loop | The hierarchical planner can either replace this or sit *inside* it as a new `solver` | Replace `solver` with a `HierarchicalSolver` that produces the same `outputs['actions']` shape | medium: action shape and `process['action']` interplay | Test: mock solver returning a fixed action sequence; assert buffer pops correctly |
| HLEWM-GOAL-ENCODING | [jepa.py](jepa.py) | `JEPA.get_cost` (lines that build `goal` dict and call `self.encode(goal)`) | Encodes `info_dict['goal']` (image) into `goal_emb` | Goal image encoding | Hierarchical planner needs to encode the *final* goal image once and pass `z_goal` to the outer CEM cost | Extract into a helper `encode_goal_image(info_dict) → z_goal (B, D_z)`; cache across CEM iterations | low | Test: assert two consecutive calls with same `goal` produce identical latents |
| HLEWM-WAYPOINT-SAMPLER | MISSING | `hlewm/data/waypoint_sampler.py: WaypointSampler` (proposed) | – | Waypoint subsampling | Stage-2 dataset needs sorted, valid waypoint indices for each trajectory | New module | medium: variable trajectory lengths | Unit tests: see §6 Step 4 |
| HLEWM-ACTION-ENCODER | MISSING | `hlewm/modules/action_encoder.py: ActionEncoder` (proposed) | – | `A_psi` (latent macro-action encoder) | Compresses action chunks → `D_l`-dim latents that condition `P^(2)` | New module | medium: padding/masking variable-length chunks | Unit tests: see §6 Step 6 |
| HLEWM-HIGH-PREDICTOR | MISSING | `hlewm/modules/high_level_predictor.py: HighLevelPredictor` (proposed; wraps `ARPredictor`) | – | `P^(2)` | Operates at waypoint spacing | New module reusing `ARPredictor` with `input_dim=D_z, action_dim=D_l, num_frames=1` | low | Unit tests: see §6 Step 7 |
| HLEWM-STAGE2-DATA | MISSING | `hlewm/data/stage2_dataset.py: Stage2Dataset` (proposed; wraps `HDF5Dataset`) | – | Stage-2 data | Yields `(waypoint_obs, action_chunks, lengths_mask)` | New module | medium: re-using HDF5Dataset's flat slice API for long episodes | Unit tests: see §6 Step 5 |
| HLEWM-STAGE2-TRAIN | MISSING | `train_stage2.py` (proposed) | – | Stage-2 training loop | Trains only `A_psi + P^(2)` | New entry point | medium | Smoke test: 1 epoch on 10 fake trajectories |
| HLEWM-FREEZE | MISSING | `hlewm/training/freeze.py` (proposed) | – | Frozen `E` + `P^(1)` | Required to preserve LeWM baseline within Stage-2 | New helper that calls `requires_grad_(False) + .eval()` on the relevant submodules | medium: must include `BatchNorm1d` running stats | Unit tests: see §6 Step 8 |
| HLEWM-OUTER-CEM | MISSING | `hlewm/planner/outer_cem.py: OuterCEM` (proposed) | – | Outer CEM over `l` | Operates entirely in latent space | New `Costable` adapter that exposes a CEM-friendly `get_cost` to `swm.solver.CEMSolver` | high (latent CEM is brittle) | Toy test: handcrafted `P^(2)` with known global min |
| HLEWM-INNER-CEM | MISSING | `hlewm/planner/inner_cem.py: InnerCEM` (proposed) | – | Inner CEM over primitive actions | Reuse existing `swm.solver.CEMSolver` with a subgoal-injected `Costable` | New `Costable` wrapper around `JEPA` that overrides the goal latent | medium | Test: assert flat CEM still works with the wrapper when `subgoal == final_goal_latent` |
| HLEWM-HIER-PLAN | MISSING | `hlewm/planner/hierarchical_planner.py: plan_hierarchical` (proposed) | – | Two-level planning | Orchestrates encode-obs → encode-goal → outer CEM → take first subgoal → inner CEM → return first primitive action | New function | high | Mock both predictors; check call order and output shape |
| HLEWM-CONFIG | [config/train](config/train), [config/eval](config/eval) | Hydra YAML configs | Stage-1 hyperparameters | – | Need new fields for Stage-2 + hierarchical planning | Add `config/train/hlewm_stage2.yaml`, `config/eval/{<env>}_hier.yaml`, and a `use_hierarchical_planning` flag in `config/eval/<env>.yaml` | medium | Test: load each config and assert defaults reproduce baseline |
| HLEWM-EVAL | [eval.py](eval.py) | `run(cfg)` | Builds flat planner | – | Choose flat vs hierarchical based on a config flag | Add a branch: if `cfg.plan_config.use_hierarchical_planning`, instantiate `HierarchicalPlanner` instead of `CEMSolver` | medium | Test: run with flag off → produces same outputs as today; flag on → exercises new path |
| HLEWM-VOE | MISSING | – | – | Violation-of-Expectation tests | LeWM paper mentions "surprise evaluation"; needed if we want VoE-style eval | Optional Step 14b | low | – |
| HLEWM-TESTS | MISSING | `tests/` (proposed) | – | – | No tests in repo today | Add `pytest` test suite covering every new module | medium | `pytest tests/hlewm -q` |

---

## 5. What Should Not Change

For the first H-LeWM implementation, the following baseline pieces
**must remain bit-for-bit identical** so that the original LeWM results
remain reproducible from this repository:

1. **Encoder architecture and weights.** `spt.backbone.utils.vit_hf(...)`
   call signature in [train.py](train.py), the `interpolate_pos_encoding`
   flag, and the loaded ViT-tiny weights. Stage-2 reads them through
   `model.encoder`; it must not re-instantiate the backbone.
2. **Original low-level predictor `P^(1)`.** `ARPredictor` in
   [module.py](module.py), the `JEPA.predictor` attribute in
   [jepa.py](jepa.py), and the `predict` method. The hierarchical planner
   must call the *same* `predict` so that the flat baseline and the
   inner-CEM-of-H-LeWM use exactly the same dynamics.
3. **SIGReg.** `SIGReg` in [module.py](module.py), the `(T,B,D)` ordering
   contract, and the `kwargs={'knots':17,'num_proj':1024}` defaults. Do
   not double-apply SIGReg to high-level latents — those are already in
   the SIGReg-regularized latent space because they are `E(o_k)`.
4. **Original LeWM training loop.** `lejepa_forward` in
   [train.py](train.py) and the corresponding `config/train/lewm.yaml`
   must still produce the existing baseline checkpoints when run alone.
5. **Original flat CEM planner.** `swm.solver.CEMSolver` must remain
   available and remain the default planner unless a config flag
   `plan_config.use_hierarchical_planning=true` is set.
6. **Original evaluation scripts.** `eval.py` must still work without
   `--config-name=*_hier` and must produce the same metric files
   (`pusht_results.txt`, etc.) for the original configs.

**Scientific motivation:** the whole point of this project is to
*compare* flat LeWM with H-LeWM-on-LeWM. If we modify the baseline (even
"to make it cleaner"), then any improvement we observe could be
attributed to incidental refactoring rather than to the hierarchical
extension. The baseline should be touchable only through the addition
of a single `if use_hierarchical_planning:` branch in [eval.py](eval.py).

---

## 6. Stage 2 Implementation Plan

Notation used throughout:
- `B` = batch size
- `T` = primitive trajectory length within a Stage-2 sample
- `N` = number of waypoints per trajectory (waypoint count)
- `D_z` = latent dimension (default 192, = ViT-tiny hidden = embed_dim)
- `D_a` = effective primitive action dimension (= `frameskip * raw_action_dim`)
- `D_l` = latent macro-action dimension (new; default 8)
- `H_high` = high-level planning horizon (default 3)
- `H_low` = low-level planning horizon (default 5; reuses LeWM's value)
- `L_chunk` = number of primitive actions between two consecutive waypoints
- `S_outer`, `S_inner` = CEM sample counts at outer/inner levels

---

### Step 1: Baseline reproducibility and smoke tests

**Goal**

Before writing any new code, confirm the existing LeWM baseline runs
end-to-end on this machine — training for one minimal epoch, evaluating
a few episodes with the flat planner, and saving an object checkpoint.
This establishes the "no change" baseline and produces the first
checkpoint that Stage-2 will consume.

**Paper link**

LeWM Section "Training" + Section "Planning".

**Files to modify**

- None permanently. Optionally add a new `tests/test_baseline_smoke.py`
  that runs a 10-step training and a 2-episode eval (skipped by default
  if `$STABLEWM_HOME` is unset).

**New code to add**

- `tests/test_baseline_smoke.py::test_training_step_runs` — instantiates
  `JEPA`, calls `lejepa_forward` on a synthetic batch (random pixels and
  actions) of shape `(B=2, T=4, 3, 224, 224)` and `(B=2, T=4, 10)`, and
  checks that the resulting `loss` is finite and has `grad_fn`.
- `tests/test_baseline_smoke.py::test_get_cost_runs` — instantiates
  `JEPA`, builds a fake `info_dict` with `pixels (B=2, S=2, T=4,
  3,224,224)`, `goal (B=2,1,3,224,224)`, `action (B=2,S=2,T=4,10)`,
  calls `model.get_cost(info, candidates)` and asserts output shape
  `(B=2, S=2)`.

**Expected tensor shapes**

- training_step:
  - `pixels: (B, T, 3, 224, 224)`
  - `action: (B, T, D_a)`
  - `emb: (B, T, D_z)`
  - `pred_emb, tgt_emb: (B, history_size, D_z)`
  - `loss: scalar`
- get_cost:
  - `candidates: (B, S, T, D_a)`
  - returned `cost: (B, S)`

**PyTorch notes**

- A scalar loss is just a 0-dim `torch.Tensor` with `grad_fn != None`.
- The smoke tests should set seeds (`torch.manual_seed`,
  `np.random.seed`) and use `torch.set_default_device('cpu')` to avoid
  GPU requirements for CI.
- Use `pytest.importorskip("stable_pretraining")` to skip cleanly if the
  optional dep isn't installed in the test env.

**Unit tests**

- `test_training_step_runs`: random batch, random model; asserts loss is
  finite and `loss.requires_grad is True`. Catches accidental detachment.
- `test_get_cost_runs`: catches shape mismatches in `JEPA.rollout` /
  `JEPA.criterion`.

**Manual sanity check**

```bash
# 1) tiny training run on PushT (assumes data downloaded under STABLEWM_HOME)
python train.py data=pusht trainer.max_epochs=1 trainer.devices=1 \
    loader.batch_size=8 loader.num_workers=0 wandb.enabled=False

# 2) evaluation with the resulting checkpoint
python eval.py --config-name=pusht.yaml policy=<hydra-job-id>/lewm \
    eval.num_eval=2 eval.eval_budget=10
```

**Acceptance criteria**

- `pytest tests/test_baseline_smoke.py -q` passes.
- A 1-epoch training run produces an `*_object.ckpt` file under
  `$STABLEWM_HOME/<hydra:job.id>/`.
- The eval run prints a non-error metrics dict.

**Rollback**

- Trivial: delete `tests/test_baseline_smoke.py`. No source files
  modified.

---

### Step 2: Isolate encoder and low-level predictor APIs

**Goal**

Wrap the LeWM submodules with stable, type-annotated helpers so the
hierarchical planner can reuse them without depending on `JEPA`'s
`info_dict` contract. Behavior is unchanged.

**Paper link**

LeWM encoder `E` and predictor `P^(1)`.

**Files to modify / create**

- Create `hlewm/__init__.py`.
- Create `hlewm/lewm_api.py` containing:
  - `encode_image(model: JEPA, image: Tensor) -> Tensor` (shape
    `(B, 3, H, W) -> (B, D_z)`).
  - `rollout_low(model: JEPA, z_history: Tensor, action_chunk: Tensor,
    history_size: int = 3) -> Tensor` returning the rolled-out latent
    trajectory.
  - `latent_cost(z_pred: Tensor, z_goal: Tensor, metric: str = 'mse') ->
    Tensor` (per-candidate cost).
- No edits to [jepa.py](jepa.py).

**New code to add**

- Pure-function helpers; no new `nn.Module` subclasses. Each helper is
  a thin shim over existing `JEPA` methods.

**Expected tensor shapes**

- `encode_image`: `(B, 3, H, W) -> (B, D_z)`.
- `rollout_low`: `z_history (B, history_size, D_z)`,
  `action_chunk (B, L, D_a)` → `(B, history_size + L, D_z)`.
- `latent_cost`: `(B, S, D_z)`, `(B, D_z)` → `(B, S)`.

**PyTorch notes**

- These helpers should be `@torch.inference_mode()`-decorated since they
  are intended for planning, not training.
- `encode_image` should accept either `(B, 3, H, W)` or `(B, T, 3, H, W)`
  and dispatch via `if image.ndim == 4: image = image.unsqueeze(1)`.
- `rollout_low` re-implements the loop in `JEPA.rollout` but with a
  cleaner contract (no `info_dict` mutation).

**Unit tests**

- `test_encode_image_single_vs_batched`: same image once vs same image
  in a batch produces identical latents.
- `test_rollout_low_consistency`: rollout for 1 step then 1 more step
  equals rollout for 2 steps (within numerical tolerance, given dropout
  off in eval mode).
- `test_latent_cost_shape`: random latents → output `(B, S)`.

**Manual sanity check**

```python
from jepa import JEPA  # constructed from a checkpoint
from hlewm.lewm_api import encode_image, rollout_low, latent_cost
z = encode_image(model, torch.randn(2, 3, 224, 224).cuda())
print(z.shape)  # (2, 192)
```

**Acceptance criteria**

- All Step 2 unit tests pass.
- Re-running `pytest tests/test_baseline_smoke.py` still passes.

**Rollback**

- Delete `hlewm/lewm_api.py`. Nothing in the baseline depends on it.

---

### Step 3: Add H-LeWM configuration flags

**Goal**

Introduce all the new config knobs in a backward-compatible way. The
default values must reproduce the baseline exactly.

**Paper link**

n/a (engineering).

**Files to modify / create**

- Create `config/train/hlewm_stage2.yaml`:

```yaml
defaults:
  - _self_
  - data: pusht

# Inherits from lewm.yaml structurally but trains only stage-2 modules
stage2:
  ckpt: ???                       # path to stage-1 checkpoint (object)
  num_waypoints: 4                # N
  latent_action_dim: 8            # D_l
  freeze_encoder_stage2: true
  freeze_low_predictor_stage2: true
  high_level_loss_type: l1        # one of {l1, l2}
  waypoint_sampling: random       # one of {random, fixed_stride, endpoints}
  fixed_stride: 5                 # only used when waypoint_sampling == fixed_stride

trainer:
  max_epochs: 30
  devices: auto
  accelerator: gpu
  precision: bf16
  gradient_clip_val: 1.0

action_encoder:
  hidden_dim: 256
  depth: 2
  heads: 4

high_predictor:
  depth: 4
  heads: 8
  dim_head: 64
  mlp_dim: 1024
  dropout: 0.0
```

- Add new fields to `config/eval/<env>.yaml` files (one example for
  PushT):

```yaml
plan_config:
  # existing fields ...
  use_hierarchical_planning: false   # default: baseline flat planner
  high_level_horizon: 3              # H_high
  low_level_horizon: 5               # H_low; equals existing horizon
  outer_cem_samples: 200             # S_outer
  inner_cem_samples: 300             # S_inner (equals existing num_samples)
```

**New code to add**

- Hydra reads the new fields automatically. No code edits needed yet —
  the flags are consumed in Steps 8, 12, 13.

**Expected tensor shapes**

- n/a (config only).

**PyTorch notes**

- Use `OmegaConf.to_container(cfg, resolve=True)` if these flags need to
  be hashed for caching.

**Unit tests**

- `test_config_loads_baseline_default`: load
  `config/eval/pusht.yaml` and assert `use_hierarchical_planning == false`.
- `test_config_loads_hier_override`: with override
  `plan_config.use_hierarchical_planning=true`, assert the override
  propagates.

**Manual sanity check**

```bash
python -c "import hydra; from hydra import compose, initialize; \
  initialize(config_path='config/eval', version_base=None); \
  cfg = compose(config_name='pusht'); print(cfg.plan_config)"
```

**Acceptance criteria**

- Configs load without errors via `hydra.compose`.
- Baseline `python eval.py --config-name=pusht.yaml` continues to work
  exactly as before.

**Rollback**

- Delete the new YAML keys; baseline reads only the old keys.

---

### Step 4: Implement waypoint sampler

**Goal**

A deterministic, seedable function that, given an episode length `T`
and a desired number of waypoints `N`, returns sorted waypoint indices
`(i_0=0, i_1, ..., i_{N-1}=T-1)` such that consecutive gaps are bounded.

**Paper link**

HWM "waypoint subsampling".

**Files to modify / create**

- `hlewm/data/waypoint_sampler.py`:
  - `def sample_waypoints(episode_length: int, num_waypoints: int,
    mode: Literal['random','fixed_stride','endpoints'], rng:
    np.random.Generator, min_gap: int = 1) -> np.ndarray`
  - Modes:
    - `endpoints`: returns only `[0, T-1]` (requires `N=2`).
    - `fixed_stride`: returns `[0, s, 2s, ..., T-1]` with `s` chosen so
      the total count == `N`.
    - `random`: returns `0`, `T-1`, and `N-2` uniformly sampled interior
      indices, sorted, with `min_gap` enforced.

**New code to add**

- A single function plus its tests.

**Expected tensor shapes**

- Output: `np.ndarray` of dtype `int64`, shape `(N,)`, sorted, with
  `out[0] == 0` and `out[-1] == episode_length - 1`.

**PyTorch notes**

- Pure NumPy is fine here; no autograd involvement.
- Use `np.random.default_rng(seed)` for reproducibility.

**Unit tests**

- `test_waypoint_endpoints`: `mode='endpoints'`, asserts
  `out == np.array([0, T-1])`.
- `test_waypoint_fixed_stride`: asserts equal gaps within `±1`.
- `test_waypoint_random_sorted_bounded`: asserts `out[0] == 0`,
  `out[-1] == T-1`, `np.all(np.diff(out) >= min_gap)`, length `== N`.
- `test_waypoint_random_reproducible`: same seed → same output across
  100 trials.

**Manual sanity check**

```python
from hlewm.data.waypoint_sampler import sample_waypoints
import numpy as np
print(sample_waypoints(50, 5, mode='random', rng=np.random.default_rng(0)))
```

**Acceptance criteria**

- All Step 4 unit tests pass.

**Rollback**

- Delete the file; no baseline dependency.

---

### Step 5: Implement stage-2 batch construction

**Goal**

Transform a long primitive trajectory into a Stage-2 sample:

- `waypoint_pixels: (N, 3, H, W)`
- `waypoint_actions_chunks: (N-1, L_max, D_a)`
- `chunk_mask: (N-1, L_max)` boolean — True for valid action steps.

**Paper link**

HWM Stage-2 data construction.

**Files to modify / create**

- `hlewm/data/stage2_dataset.py`:
  - `class Stage2Dataset(torch.utils.data.Dataset)`:
    - `__init__(self, base: swm.data.HDF5Dataset, num_waypoints: int,
      waypoint_sampling: str, fixed_stride: int | None = None, seed:
      int = 0)`.
    - `__getitem__(idx)` returns the dict above.
  - `def stage2_collate(batch)` returns batched tensors with consistent
    `L_max` (the max chunk length within the batch).

**New code to add**

- New `Dataset` and collator. Loads full episodes by calling
  `base.load_episode(ep_idx)` (already exists on `swm.data.Dataset`).

**Expected tensor shapes**

- `waypoint_pixels: (B, N, 3, H, W)`
- `waypoint_actions_chunks: (B, N-1, L_max, D_a)`
- `chunk_mask: (B, N-1, L_max)`

**PyTorch notes**

- The HDF5 dataset's `__getitem__` returns *short* slices (4 steps); for
  Stage-2 we need *full* episodes, so use `load_episode(ep_idx)`
  directly. The mapping from a Stage-2 dataset index to an episode
  index is one-to-one (or one-to-K if multiple subsamples per episode).
- Use `torch.nn.utils.rnn.pad_sequence` for the chunk dim or do manual
  zero-padding plus a separate mask tensor.

**Unit tests**

- `test_stage2_dataset_shapes`: synthetic dataset with 3 episodes of
  length 20; assert shapes for `N=5`.
- `test_stage2_collate_padding`: two samples with different `L_max`,
  assert post-collation shapes use the global max.
- `test_stage2_dataset_reproducible`: same seed → same waypoints.

**Manual sanity check**

```python
ds = Stage2Dataset(base_ds, num_waypoints=4, waypoint_sampling='random', seed=0)
sample = ds[0]
print({k: v.shape for k, v in sample.items()})
```

**Acceptance criteria**

- All Step 5 unit tests pass.
- Dry-run iteration of a `DataLoader(stage2_dataset, batch_size=4)`
  completes without error.

**Rollback**

- Delete the new dataset module.

---

### Step 6: Implement `ActionEncoder` (`A_psi`)

**Goal**

A small Transformer encoder that takes a (possibly padded) chunk of
primitive actions and returns a single `D_l`-dim latent macro-action,
respecting the padding mask.

**Paper link**

HWM action encoder `A_psi`.

**Files to modify / create**

- `hlewm/modules/action_encoder.py`:
  - `class ActionEncoder(nn.Module)`:
    - `__init__(self, action_dim: int, latent_dim: int, hidden_dim:
      int = 256, depth: int = 2, heads: int = 4)`.
    - Components: a `nn.Linear(action_dim, hidden_dim)` input proj, a
      learnable `cls_token` of shape `(1, 1, hidden_dim)`, learnable
      positional embeddings, `depth` plain `Block` layers from
      [module.py](module.py) (no causal mask), a final
      `nn.Linear(hidden_dim, latent_dim)` output proj.
    - `forward(actions: Tensor, mask: BoolTensor) -> Tensor`.

**New code to add**

- One class plus its test file.

**Expected tensor shapes**

- Input: `actions (B*N, L_max, D_a)` and `mask (B*N, L_max)` (True =
  valid).
- Output: `(B*N, D_l)`; reshape to `(B, N, D_l)` after.

**PyTorch notes**

- The `Block` class in [module.py](module.py) does **not** accept an
  attention mask; we will need to either (a) write a new `Attention`
  variant that supports `key_padding_mask`, or (b) implement it directly
  here with `F.scaled_dot_product_attention(..., attn_mask=...)`. Path
  (b) is recommended to avoid touching baseline code.
- Use a `[CLS]` token prepended at index 0; pool by taking the CLS
  output. The mask must be extended by 1 (CLS is always valid).
- Set `is_causal=False`; this is bidirectional.

**Unit tests**

- `test_action_encoder_shape`: `B*N=8, L_max=12, D_a=10, D_l=8` →
  output `(8, 8)`.
- `test_action_encoder_mask_invariance`: padding with zeros vs random
  values past the mask should produce the **same** output (proves the
  mask is being used).
- `test_action_encoder_gradients`: `loss = encoder(...).sum().backward()`,
  assert all parameters have non-None grads.
- `test_action_encoder_eval_deterministic`: in `.eval()` mode, two calls
  with the same input give identical output.

**Manual sanity check**

```python
ae = ActionEncoder(action_dim=10, latent_dim=8)
acts = torch.randn(2, 12, 10); mask = torch.ones(2, 12, dtype=torch.bool); mask[:, 8:] = False
print(ae(acts, mask).shape)  # (2, 8)
```

**Acceptance criteria**

- All Step 6 tests pass.

**Rollback**

- Delete `hlewm/modules/action_encoder.py`.

---

### Step 7: Implement high-level predictor `P^(2)`

**Goal**

A module that maps `(z_k ∈ R^{D_z}, l_k ∈ R^{D_l})` to
`ẑ_{k+1} ∈ R^{D_z}` and supports autoregressive rollout over `H_high`
steps.

**Paper link**

HWM high-level predictor `P^(2)`.

**Files to modify / create**

- `hlewm/modules/high_level_predictor.py`:
  - `class HighLevelPredictor(nn.Module)`:
    - Internally instantiates `module.ARPredictor(num_frames=H_high,
      input_dim=D_z, hidden_dim=D_z, output_dim=D_z, depth=4, heads=8,
      mlp_dim=1024, dim_head=64, dropout=0.0)`.
    - It also instantiates a tiny `nn.Linear(D_l, D_z)` because
      `ARPredictor` expects `c` to be the same dim as `input_dim` (see
      `cond_proj` in [module.py](module.py)).
    - `forward(z: Tensor, l: Tensor) -> Tensor` (teacher-forced
      one-shot).
    - `rollout(z_init: Tensor, l_seq: Tensor) -> Tensor` (autoregressive
      across `H_high` steps).

**New code to add**

- One class plus tests.

**Expected tensor shapes**

- `forward`:
  - `z (B, H_high, D_z)`, `l (B, H_high, D_l)`
  - returns `(B, H_high, D_z)` (predicts each `z_{k+1}` from `z_k, l_k`).
- `rollout`:
  - `z_init (B, D_z)`, `l_seq (B, H_high, D_l)`
  - returns `(B, H_high, D_z)` = the rolled-out `[ẑ_1, ẑ_2, ..., ẑ_{H_high}]`.

**PyTorch notes**

- The simplest variant treats each step as Markov: predict `ẑ_{k+1}`
  from `(z_k, l_k)` alone. Then `num_frames=1` in `ARPredictor` and
  `rollout` is a Python loop that keeps only the last predicted latent.
- Alternative: maintain a high-level history of size `H_history=3`.
  Defer this to an ablation.

**Unit tests**

- `test_high_predictor_single_step_shape`: random inputs → output of
  correct shape.
- `test_high_predictor_rollout_shape`: `z_init (4, 192)`, `l_seq (4, 3,
  8)` → output `(4, 3, 192)`.
- `test_high_predictor_rollout_equals_step_by_step`: rollout for 3
  steps matches three manual single-step calls (deterministic, in
  eval mode).

**Manual sanity check**

```python
hp = HighLevelPredictor(D_z=192, D_l=8)
z0 = torch.randn(2, 192); ls = torch.randn(2, 3, 8)
print(hp.rollout(z0, ls).shape)  # (2, 3, 192)
```

**Acceptance criteria**

- All Step 7 tests pass.

**Rollback**

- Delete `hlewm/modules/high_level_predictor.py`.

---

### Step 8: Implement stage-2 high-level loss

**Goal**

Compute the teacher-forced high-level loss:

```
z_k       = E(o_k)                  for k = 0..N-1   (frozen)
l_k       = A_psi(a_chunk_k, mask_k) for k = 0..N-2
ẑ_{k+1}   = P^(2)(z_k, l_k)         for k = 0..N-2
L_high    = mean_k  ||ẑ_{k+1} - z_{k+1}||_1
```

**Paper link**

HWM Stage-2 loss (L1).

**Files to modify / create**

- `hlewm/training/losses.py`:
  - `def high_level_loss(model: JEPA, action_encoder: ActionEncoder,
    high_predictor: HighLevelPredictor, batch: dict, loss_type: str
    = 'l1') -> tuple[Tensor, dict]`.
- `hlewm/training/freeze.py`:
  - `def freeze_lewm(model: JEPA) -> None`:
    - `for p in model.encoder.parameters(): p.requires_grad_(False)`.
    - `model.encoder.eval()`.
    - `for p in model.projector.parameters(): p.requires_grad_(False)`.
    - `model.projector.eval()`.  # BatchNorm running stats
    - similarly for `predictor`, `pred_proj`, `action_encoder` (the
      *primitive* one).

**New code to add**

- Two helpers + tests.

**Expected tensor shapes**

- `waypoint_pixels (B, N, 3, H, W)` → after `E`: `(B, N, D_z)`.
- `action_chunks (B, N-1, L_max, D_a)`, mask `(B, N-1, L_max)` → after
  `A_psi`: `(B, N-1, D_l)`.
- `z_curr = z[:, :-1] (B, N-1, D_z)`, `z_next = z[:, 1:] (B, N-1, D_z)`.
- `pred (B, N-1, D_z)` from `P^(2)(z_curr, l)`.
- `loss = (pred - z_next).abs().mean()` — scalar.

**PyTorch notes**

- `with torch.no_grad():` is *not* sufficient on its own when modules
  contain BatchNorm — calling `.eval()` is required to stop running-stat
  updates. Both are required.
- The encoder forward call should still propagate to compute features,
  but the resulting tensors should have `requires_grad=False`. This is
  automatic when all encoder params are frozen and the input tensor has
  `requires_grad=False`.
- It's safe to call `z_curr.detach()` for extra safety.

**Unit tests**

- `test_freeze_affects_only_lewm`: after `freeze_lewm(model)`, walk all
  modules and assert that `encoder/projector/predictor/pred_proj/
  action_encoder(primitive)` have `requires_grad=False` and are in
  `.training==False`, while a newly added `HighLevelPredictor` and the
  new `ActionEncoder` still have `requires_grad=True`.
- `test_high_loss_scalar_finite`: random `JEPA` + random Stage-2 batch
  → `loss` is finite scalar.
- `test_high_loss_gradients_only_to_new_modules`: backprop through
  `loss` and assert `model.encoder.parameters` have grad None, while
  `high_predictor.parameters` and `action_encoder.parameters` have
  non-None grads.
- `test_high_loss_shape_mismatch_raises`: if `action_chunks` has wrong
  `N`, expect a clear error.

**Manual sanity check**

```python
loss, logs = high_level_loss(model, ae, hp, fake_batch)
loss.backward()
assert next(model.encoder.parameters()).grad is None
```

**Acceptance criteria**

- All Step 8 tests pass.

**Rollback**

- Delete `hlewm/training/losses.py` and `freeze.py`.

---

### Step 9: Implement stage-2 training loop

**Goal**

A new entry point `train_stage2.py` that:

1. Loads the Stage-1 LeWM object checkpoint.
2. Builds the Stage-2 dataset and DataLoader.
3. Constructs `ActionEncoder` and `HighLevelPredictor`.
4. Freezes `E` and `P^(1)` (Step 8 helper).
5. Trains only the new modules with the high-level L1 loss.
6. Saves a new object checkpoint of an `nn.Module` wrapper that exposes
   both the old `get_cost` (delegated to the LeWM `JEPA`) and a new
   `get_cost_hier` method usable by the hierarchical planner.

**Paper link**

HWM Stage-2 procedure.

**Files to modify / create**

- `train_stage2.py` (Hydra `config_name=hlewm_stage2`).
- `hlewm/models/hlewm.py`:
  - `class HLeWM(nn.Module)` that holds `lewm: JEPA`, `action_encoder:
    ActionEncoder`, `high_predictor: HighLevelPredictor`, and exposes:
    - `def get_cost(self, info_dict, action_candidates) -> Tensor`
      (delegate to `self.lewm.get_cost`).
    - `def get_cost_hier(self, info_dict, macro_action_candidates) ->
      Tensor` (Step 11 / 12 will use this).

**New code to add**

- Stage-2 entry point modeled on [train.py](train.py), with the
  following key differences:
  - Optimizer is built from
    `list(action_encoder.parameters()) +
    list(high_predictor.parameters())` only.
  - Forward function calls `freeze_lewm(self.lewm)` once and
    `high_level_loss(...)` per step.

**Expected tensor shapes**

- Same as Step 8.

**PyTorch notes**

- Calling `freeze_lewm` once in `__init__` is enough. Lightning calls
  `module.train()` at the start of every training epoch which would
  flip the frozen modules back to training mode and re-enable BN
  running-stat updates. Workaround: override
  `train(self, mode=True)` on the `HLeWM` wrapper to keep frozen
  submodules in `.eval()`. This is the standard "freeze pattern" idiom.
- Use the same `LinearWarmupCosineAnnealingLR` schedule for parity.
- Object checkpoint must be saved with the `lewm` submodule **still
  inside** so that `swm.policy.AutoCostModel` can find `get_cost`.

**Unit tests**

- `test_stage2_forward_runs`: tiny fake dataset, build `HLeWM`, run one
  training step, assert loss is finite.
- `test_stage2_only_new_params_have_grad`: assertion same as Step 8.
- `test_stage2_checkpoint_roundtrip`: `torch.save(hlewm, path)` then
  `torch.load(path, weights_only=False)`, assert
  `loaded.lewm.predictor.pos_embedding` is bit-equal to the original.
- `test_stage2_train_method_keeps_frozen_eval`: after
  `hlewm.train()`, `hlewm.lewm.encoder.training == False`.

**Manual sanity check**

```bash
python train_stage2.py stage2.ckpt=<job-id>/lewm_epoch_1 \
    trainer.max_epochs=1 loader.batch_size=4 loader.num_workers=0 \
    wandb.enabled=False
```

**Acceptance criteria**

- All Step 9 tests pass.
- A 1-epoch Stage-2 run produces an object checkpoint that can be
  loaded by `swm.policy.AutoCostModel(...)` and exposes both `get_cost`
  (flat) and `get_cost_hier`.

**Rollback**

- Delete `train_stage2.py` and `hlewm/models/hlewm.py`. The baseline
  Stage-1 path is untouched.

---

### Step 10: Implement outer CEM over latent macro-actions

**Goal**

A CEM optimizer over macro-action sequences. Given:
- current encoded latent `z_curr (B, D_z)`,
- goal latent `z_goal (B, D_z)`,
- high-level horizon `H_high`,
- macro-action dim `D_l`,
- sample count `S_outer`,
return:
- the best macro-action sequence `l* (B, H_high, D_l)`,
- the predicted subgoal latents `ẑ* (B, H_high, D_z)` (especially `ẑ_1*`).

**Paper link**

HWM outer planner.

**Files to modify / create**

- `hlewm/planner/outer_cem.py`:
  - `class OuterCEMCost`:
    - Wraps `HighLevelPredictor` so that calling
      `get_cost(info, candidates)` returns `(B, S_outer)`.
    - `info` carries `z_curr (B, D_z)` and `z_goal (B, D_z)`.
    - `candidates (B, S_outer, H_high, D_l)`.
  - Reuses `swm.solver.CEMSolver` with a faked
    `gym.spaces.Box(low=-inf, high=inf, shape=(1, D_l))` action space
    and `action_block=1`.
- Alternative: implement a small CEM directly in NumPy/Torch (it is
  ~60 lines).

**New code to add**

- `OuterCEMCost` adapter + a thin `run_outer_cem(...)` function.

**Expected tensor shapes**

- Input `info`: `{"z_curr": (B, D_z), "z_goal": (B, D_z)}`.
- Candidates: `(B, S_outer, H_high, D_l)` sampled from
  `N(mean (B, H_high, D_l), var (B, H_high, D_l))`.
- Rollout: `(B*S_outer, H_high, D_z)` after flatten.
- Cost per candidate: `||ẑ_{H_high} - z_goal||_2^2` (or L1; configurable).
  Returned shape `(B, S_outer)`.
- Output: `l* (B, H_high, D_l)`, `ẑ* (B, H_high, D_z)`.

**PyTorch notes**

- Macro-actions are continuous, unbounded, and have no domain
  meaning. Initialize `mean=0, var=1` and rely on SIGReg-trained
  latents to live in a "small" region around 0.
- Do not apply any `process['action'].inverse_transform` to the
  macro-actions — those normalizers are for primitive env actions.
- Decorate the planning function with `@torch.inference_mode()`.

**Unit tests**

- `test_outer_cem_toy_quadratic`: replace `HighLevelPredictor` with a
  fake linear dynamics `ẑ_{k+1} = z_k + l_k`. With `z_goal = z_curr +
  k_star` for some fixed `k_star`, the optimal `l_0 + l_1 + ... ==
  k_star`. Assert convergence to within tolerance.
- `test_outer_cem_shapes`: random predictor, asserts output shapes.
- `test_outer_cem_decreasing_cost`: cost at step 0 ≥ cost at step
  `n_steps-1` (monotonic-ish).

**Manual sanity check**

```python
out = run_outer_cem(hp, z_curr=torch.randn(2,192), z_goal=torch.randn(2,192),
                   H_high=3, D_l=8, S_outer=200, n_iters=30)
print(out['l_star'].shape, out['z_star'].shape)
```

**Acceptance criteria**

- All Step 10 tests pass.

**Rollback**

- Delete `hlewm/planner/outer_cem.py`.

---

### Step 11: Implement inner CEM over primitive actions

**Goal**

Re-use the existing flat CEM solver but with the *subgoal latent*
(returned by outer CEM) as the planning target instead of the encoded
goal image.

**Paper link**

HWM inner planner.

**Files to modify / create**

- `hlewm/planner/inner_cem.py`:
  - `class SubgoalCostModel(nn.Module)`:
    - `__init__(self, lewm: JEPA)`.
    - `forward(...)`: trivial.
    - `get_cost(self, info_dict, action_candidates) -> Tensor`:
      same body as `JEPA.get_cost` but reads `info_dict['z_subgoal']`
      directly (a tensor) instead of encoding `info_dict['goal']`.
  - `def run_inner_cem(lewm: JEPA, info_dict: dict, z_subgoal: Tensor,
    horizon: int, n_samples: int) -> dict`:
    - Creates a `CEMSolver(model=SubgoalCostModel(lewm), ...)` once.
    - Calls `.configure(action_space=..., n_envs=B, config=...)`.
    - Returns `solver.solve(info_dict)`.

**New code to add**

- `SubgoalCostModel` and `run_inner_cem` + tests.

**Expected tensor shapes**

- `info_dict['pixels']`: `(B, T_obs=1, 3, H, W)` (or whatever the planner
  feeds in).
- `info_dict['z_subgoal']`: `(B, D_z)`.
- `action_candidates`: `(B, S_inner, H_low, D_a)`.
- Output `cost`: `(B, S_inner)`.

**PyTorch notes**

- `swm.solver.CEMSolver.configure(...)` mutates internal attributes; do
  this once per outer call (not once per CEM iteration).
- The trickiest piece is that `JEPA.get_cost` currently re-encodes goal
  on every call; `SubgoalCostModel.get_cost` must skip that and inject
  `z_subgoal` into the broadcast-expanded form `(B, S, ..., D_z)`.

**Unit tests**

- `test_inner_cem_equiv_to_flat_when_subgoal_equals_goal`: build a
  random model, pick a random image `o`, set `z_subgoal = E(o)`. Run
  flat CEM with `goal=o` and inner CEM with `z_subgoal`. The action
  means returned should be within a small tolerance.
- `test_inner_cem_shapes`: synthetic batch, assert returned action
  tensor shape `(B, H_low, D_a)`.

**Manual sanity check**

```python
out = run_inner_cem(lewm, info_dict, z_subgoal, horizon=5, n_samples=300)
print(out['actions'].shape)  # (B, 5, D_a)
```

**Acceptance criteria**

- All Step 11 tests pass.

**Rollback**

- Delete `hlewm/planner/inner_cem.py`.

---

### Step 12: Implement hierarchical MPC wrapper

**Goal**

The full hierarchical planner. Implements:

```
encode_current_obs:  z_curr  = E(o_t)
encode_final_goal:   z_goal  = E(g)        (cached across replans)
outer CEM:           l*, ẑ*  = run_outer_cem(hp, z_curr, z_goal, H_high, D_l, S_outer)
pick first subgoal:  z_sub   = ẑ*[:, 0]
inner CEM:           a*      = run_inner_cem(lewm, info, z_sub, H_low, S_inner)
execute first prim:  return a*[:, 0]
```

**Paper link**

HWM joint planning loop.

**Files to modify / create**

- `hlewm/planner/hierarchical_planner.py`:
  - `class HierarchicalSolver`:
    - Implements the `swm.solver.solver.Solver` protocol so it can be
      dropped into `WorldModelPolicy` in place of `CEMSolver`.
    - `configure(action_space, n_envs, config)`.
    - `solve(info_dict, init_action=None) -> dict` returning
      `outputs['actions'] (n_envs, horizon, action_dim)` where
      `horizon == receding_horizon == 1` (we re-plan every step).
  - Internally calls Step 10 + Step 11.

**New code to add**

- The hierarchical solver class plus tests.

**Expected tensor shapes**

- `info_dict['pixels']: (n_envs, T_obs, 3, H, W)`.
- `info_dict['goal']: (n_envs, T_obs, 3, H, W)`.
- `z_curr (n_envs, D_z)`, `z_goal (n_envs, D_z)`.
- Outer output: `l* (n_envs, H_high, D_l)`, `ẑ* (n_envs, H_high, D_z)`.
- Inner output: `a* (n_envs, H_low * action_block, D_raw_action)`.
- `outputs['actions']: (n_envs, 1, action_block * D_raw_action)` (take
  only the first primitive).

**PyTorch notes**

- `WorldModelPolicy.get_action` expects `solver.solve(info, init=...)`
  to return a dict with key `'actions'`. The hierarchical solver
  should respect that contract.
- Caching the goal latent across CEM iterations within a single
  `solve(...)` call is cheap; caching across calls is harder because
  the goal image can change between env steps. A reasonable
  approximation is to cache by `id(info_dict['goal'])` for the
  duration of a single episode.

**Unit tests**

- `test_hier_solver_call_order`: mock `HighLevelPredictor` and
  `JEPA` so that we can record the order of calls. Assert:
  encode-obs → encode-goal → outer-CEM-rollout (≥1 call) → inner-CEM-rollout
  → return.
- `test_hier_solver_output_shape`: synthetic everything, assert
  `outputs['actions']` matches the protocol expected by
  `WorldModelPolicy`.
- `test_hier_solver_handles_n_envs`: `n_envs=2, 5, 50`.

**Manual sanity check**

```python
hs = HierarchicalSolver(hlewm, H_high=3, D_l=8, H_low=5,
                        S_outer=100, S_inner=200)
hs.configure(action_space=env.action_space, n_envs=2, config=plan_cfg)
out = hs.solve(info_dict)
print(out['actions'].shape)
```

**Acceptance criteria**

- All Step 12 tests pass.

**Rollback**

- Delete `hlewm/planner/hierarchical_planner.py`.

---

### Step 13: Preserve flat LeWM planning as baseline

**Goal**

Add a single branch in [eval.py](eval.py) so that the same script can
launch flat CEM or the hierarchical planner based on a config flag.

**Paper link**

n/a (engineering).

**Files to modify**

- [eval.py](eval.py): in `run(cfg)`, after building `model`, branch:

```python
if cfg.plan_config.get('use_hierarchical_planning', False):
    solver = HierarchicalSolver(model, **hier_kwargs)
else:
    solver = hydra.utils.instantiate(cfg.solver, model=model)
```

That is the only change required to the baseline script.

**New code to add**

- A few lines + import. The default `use_hierarchical_planning` is
  `False`, so unchanged commands continue to work.

**Expected tensor shapes**

- Same as before; both branches return a `Solver`-protocol object that
  `WorldModelPolicy` consumes uniformly.

**PyTorch notes**

- The `Solver` protocol is in `swm.solver.solver.Solver` (runtime-checkable).
  Make sure `HierarchicalSolver` satisfies `isinstance(hs, Solver)`.

**Unit tests**

- `test_eval_baseline_unchanged`: with `use_hierarchical_planning=False`,
  the constructed solver is `swm.solver.CEMSolver`.
- `test_eval_hier_selected`: with the flag on, the solver is
  `HierarchicalSolver` and `isinstance(solver, Solver)` is True.

**Manual sanity check**

```bash
# baseline
python eval.py --config-name=pusht.yaml policy=<run>/lewm \
    eval.num_eval=2 eval.eval_budget=10
# hierarchical
python eval.py --config-name=pusht.yaml policy=<run>/hlewm \
    plan_config.use_hierarchical_planning=true \
    eval.num_eval=2 eval.eval_budget=10
```

**Acceptance criteria**

- Both commands run without error.
- The baseline command produces metrics indistinguishable from before
  this change (same seed).

**Rollback**

- Revert the if-branch and the import; baseline returns instantly.

---

### Step 14: Evaluation plan

**Goal**

Quantitatively compare flat LeWM with H-LeWM on the four supported
environments, with controls for budget and horizon.

**Paper link**

LeWM benchmarks (PushT, OGBench-Cube, Two-Room, Reacher) +
HWM long-horizon analysis.

**Metrics to log per run.**

- Success rate (already produced by `world.evaluate_from_dataset`).
- Planning wall-clock time (use `time.time()` around `solver.solve`).
- Number of CEM samples (configured) and number of forward passes through
  `P^(1)` and `P^(2)`.
- Latent rollout error vs horizon: for held-out trajectories, compute
  `||ẑ_t - z_t||` for `t=1..T` and plot.
- Optional: violation-of-expectation scores (LeWM paper Section
  "Surprise"). MISSING in this repo; can be deferred.

**Run matrix.**

| Env | Planner | Horizon (env steps) | Eval episodes |
|---|---|---|---|
| PushT | flat | 50 | 50 |
| PushT | hier | 50 | 50 |
| PushT | flat | 100 | 50 |
| PushT | hier | 100 | 50 |
| Two-Room | flat | 50 | 50 |
| Two-Room | hier | 50 | 50 |
| Two-Room | flat | 100 | 50 |
| Two-Room | hier | 100 | 50 |
| OGB-Cube | flat | 50 | 50 |
| OGB-Cube | hier | 50 | 50 |
| Reacher | flat | 50 | 50 |
| Reacher | hier | 50 | 50 |

Two-Room with `eval.eval_budget=100` is the most likely place to see
hierarchical gains, because the env has nontrivial sub-goal structure.

**Avoiding overclaiming.**

- Always report flat with the *same compute budget* (forward passes) as
  hierarchical, not the same `num_samples`.
- Report variance over ≥3 seeds.
- Do not cherry-pick `goal_offset_steps`.
- Report negative results clearly if the hierarchical planner is *not*
  better; this is a real research outcome.

**Acceptance criteria.**

- A reproducible script `scripts/run_eval_matrix.sh` (proposed) executes
  the full matrix.
- A small report (`plans/results.md`, proposed) summarizes findings.

---

### Step 15: Ablation plan

| Ablation | Values | What it tests |
|---|---|---|
| `num_waypoints` | 2, 3, 5, 8 | Sensitivity of `P^(2)` to waypoint density |
| `latent_action_dim` | 2, 4, 8, 16 | Whether `D_l` is too small to compress 5-15-step action chunks |
| Waypoint sampling mode | `random`, `fixed_stride`, `endpoints` | Robustness to waypoint selection |
| High-level loss | `l1` vs `l2` | Sensitivity to outlier latents under SIGReg |
| Frozen encoder | True (default) vs False (small LR for `E`) | Whether fine-tuning helps; risk: drift from SIGReg distribution |
| `outer_cem_samples` | 50, 100, 200, 400 | Compute–success tradeoff |
| `inner_cem_samples` | 100, 300, 600 | Inner CEM compute scaling |
| `H_high` | 2, 3, 5 | Effective lookahead |

Each row produces a separate run; record results in `plans/results.md`.

---

### Step 16: Documentation and learning notes

**Goal**

Make this project understandable to a future re-reader (you, rusty
again at PyTorch).

**Files to modify / create**

- `README.md`: add a section "## H-LeWM" with a short summary and links
  to `plans/plan.md` and the Stage-2 commands.
- `plans/learning_notes.md` (proposed): per-step "what I learned",
  including the SIGReg surprise (random projections, Epps-Pulley) and
  the AdaLN-zero pattern in `ConditionalBlock`.
- Docstrings on every new public function/class.

**Acceptance criteria.**

- A new contributor can run baseline + Stage-2 by reading `README.md`
  alone.

---

## 7. Test Matrix

| Test | Level | File | What It Checks | Fake Inputs | Expected Result | Related Step |
|---|---|---|---|---|---|---|
| `test_training_step_runs` | smoke | `tests/test_baseline_smoke.py` | Stage-1 forward produces finite scalar loss with grad_fn | random `(B=2,T=4,3,224,224)` + random actions `(B=2,T=4,10)` | `loss.item()` finite, `loss.requires_grad==True` | Step 1 |
| `test_get_cost_runs` | smoke | `tests/test_baseline_smoke.py` | `JEPA.get_cost` returns `(B,S)` | random images + candidates `(B=2,S=2,T=4,10)` | shape `(2,2)`, finite | Step 1 |
| `test_encode_image_single_vs_batched` | unit | `tests/hlewm/test_lewm_api.py` | Wrapper consistent across batch dims | one image vs same image batched | identical latents | Step 2 |
| `test_rollout_low_consistency` | unit | `tests/hlewm/test_lewm_api.py` | Two 1-step rollouts == one 2-step rollout | random | within `1e-5` | Step 2 |
| `test_waypoint_endpoints` | unit | `tests/hlewm/test_waypoint_sampler.py` | Endpoints mode returns `[0, T-1]` | `T=10, N=2` | `[0, 9]` | Step 4 |
| `test_waypoint_fixed_stride` | unit | same | Equal gaps within ±1 | `T=20, N=5` | gaps in `{4,5}` | Step 4 |
| `test_waypoint_random_sorted_bounded` | unit | same | Sorted, valid, includes endpoints | `T=50, N=5, min_gap=2` | all properties | Step 4 |
| `test_waypoint_random_reproducible` | unit | same | Same seed → same output | seed=0 | identical 100x | Step 4 |
| `test_stage2_dataset_shapes` | unit | `tests/hlewm/test_stage2_dataset.py` | Output shapes correct | synthetic episodes | `(N,3,H,W)` etc. | Step 5 |
| `test_stage2_collate_padding` | unit | same | Pads to global L_max | two samples diff lengths | unified L_max | Step 5 |
| `test_action_encoder_shape` | unit | `tests/hlewm/test_action_encoder.py` | Output `(B*N, D_l)` | `(8,12,10)` + mask | `(8, 8)` | Step 6 |
| `test_action_encoder_mask_invariance` | unit | same | Pad-content invariance | mask half | identical output | Step 6 |
| `test_action_encoder_gradients` | unit | same | All params get grads | sum loss | all `.grad` non-None | Step 6 |
| `test_high_predictor_single_step_shape` | unit | `tests/hlewm/test_high_predictor.py` | `(B, D_z) -> (B, D_z)` | random | match | Step 7 |
| `test_high_predictor_rollout_shape` | unit | same | Multi-step shape | `(4,192), (4,3,8)` | `(4,3,192)` | Step 7 |
| `test_high_predictor_rollout_equals_step_by_step` | unit | same | Deterministic equivalence | eval mode | equal within `1e-6` | Step 7 |
| `test_freeze_affects_only_lewm` | unit | `tests/hlewm/test_freeze.py` | Freeze helper only freezes LeWM | – | per-module assertions | Step 8 |
| `test_high_loss_scalar_finite` | unit | `tests/hlewm/test_high_loss.py` | Loss is finite scalar | fake batch | OK | Step 8 |
| `test_high_loss_gradients_only_to_new_modules` | unit | same | Only new modules accumulate gradients | backward | encoder.grad None | Step 8 |
| `test_stage2_forward_runs` | smoke | `tests/hlewm/test_stage2_train.py` | One forward step works | tiny synthetic | finite loss | Step 9 |
| `test_stage2_checkpoint_roundtrip` | unit | same | torch.save/load preserves params | dummy | bit-equal | Step 9 |
| `test_stage2_train_method_keeps_frozen_eval` | unit | same | Override of `train()` works | call `.train()` | encoder.training==False | Step 9 |
| `test_outer_cem_toy_quadratic` | unit | `tests/hlewm/test_outer_cem.py` | Convergence on toy linear dynamics | hand-crafted predictor | low residual | Step 10 |
| `test_outer_cem_shapes` | unit | same | Output shapes | random | `(B, H_high, D_l)`, `(B, H_high, D_z)` | Step 10 |
| `test_inner_cem_equiv_to_flat_when_subgoal_equals_goal` | unit | `tests/hlewm/test_inner_cem.py` | Subgoal-mode == goal-mode when equal | seeded | mean action within tol | Step 11 |
| `test_hier_solver_call_order` | unit | `tests/hlewm/test_hier_planner.py` | Encode → outer → inner ordering | mocks | correct order | Step 12 |
| `test_hier_solver_output_shape` | unit | same | Solver protocol satisfied | random | matches `Solver` | Step 12 |
| `test_eval_baseline_unchanged` | unit | `tests/test_eval_switch.py` | Default flag picks flat CEM | hydra compose | `isinstance(...CEMSolver)` | Step 13 |
| `test_eval_hier_selected` | unit | same | Flag picks hierarchical | override | `isinstance(...HierarchicalSolver)` | Step 13 |

---

## 8. Risks and Mitigations

| Risk | Why it matters | Mitigation | Test / Diagnostic |
|---|---|---|---|
| Unreachable latent subgoals | Outer CEM proposes a `ẑ_1*` that no primitive action sequence can reach within `H_low` steps; inner CEM oscillates | Use a *reachability check* (compute inner CEM cost at `ẑ_1*`; if above threshold, fall back to flat CEM for this step) | Log inner-CEM final cost; histogram across episodes |
| High-level predictor drift | `P^(2)` outputs latents off the SIGReg distribution → inner CEM cost is uninformative | Add an optional reg term `||ẑ_{H_high}|| ≤ τ` during Stage-2 training; or project predicted latents through `model.projector` again | Log mean / std of predicted latents vs encoded latents |
| Freezing vs fine-tuning | Allowing `E` to update in Stage-2 may give short-term gains at the cost of breaking the LeWM baseline | Default to fully frozen. Ablation only with a tiny LR (`1e-6`) on `E` and clear before/after metrics | `test_freeze_affects_only_lewm` |
| SIGReg distribution assumptions | If we naïvely L2-distance latents, isotropy implies *every* direction is equally informative; that's not what the env cares about | Try L1 (matches HWM); consider whitening or a learned cost head later | Compare L1 vs L2 ablation |
| Latent distance ≠ env success | Standard JEPA pitfall | Always report environment success rate, never latent distance alone | The eval harness already does this |
| CEM compute explosion | `S_outer * S_inner * n_iters` forward passes per step | Default to `S_outer=200, S_inner=300, n_iters=30`; profile and reduce iter counts as needed | Wall-clock logging |
| Variable-length chunks | Off-by-one or wrong mask → silent bugs in `A_psi` | Strict shape assertions; mask-invariance test | Step 6 `test_action_encoder_mask_invariance` |
| Shape/device bugs | The codebase moves between CPU/GPU at several points; `nan_to_num` is used on actions | Single device contract: all tensors in the hierarchical pipeline live on `model.device` | Add a `assert tensor.device == self.device` block at the start of each new `get_cost*` |
| Evaluation too short | LeWM eval defaults to `eval_budget=50` which may hide long-horizon gains | Run an extended-budget eval matrix (Step 14) with `eval_budget ∈ {50, 100, 150}` | Step 14 |
| Stage-1 checkpoint not produced | Stage-2 cannot start without LeWM weights | Document `train.py` command; provide a 1-epoch tiny checkpoint for tests | `tests/test_baseline_smoke.py` |
| BatchNorm running stats updated despite freezing | `projector`/`pred_proj` use BatchNorm1d; their running stats would drift if not in `.eval()` | `freeze_lewm` also calls `.eval()` on each; override `train()` on the wrapper to keep them in eval | Step 8 `test_freeze_affects_only_lewm` |

---

## 9. Open Questions

1. **Module ownership.** Should new H-LeWM modules live in a new
   top-level package `hlewm/` (recommended) or be appended to
   [module.py](module.py)? The plan assumes `hlewm/`.
2. **`P^(2)` history.** Single-state Markov (history_size=1, simplest)
   or 3-state history (parallels LeWM)? Plan starts with Markov.
3. **Latent dimension `D_z`.** Confirmed `192` (matches ViT-tiny). Verify
   no place hard-codes a different number.
4. **Action dimensions per env.** Need empirical values. From
   `config/train/data/<env>.yaml`: PushT raw `action_dim=2 → D_a=10`;
   DMC Reacher `D_a=10` (2 × frameskip 5); OGBCube `D_a=` 5 × raw
   (likely 4 or 5; confirm from data); Two-Room `D_a=10` (2 × 5).
5. **Trajectory lengths.** Are existing datasets long enough to support
   `N=5` waypoints with non-trivial inter-waypoint chunks? Need to
   query `dataset.lengths` and report `min`, `median`, `max` per env.
6. **Existing long-horizon variants.** Are there longer datasets
   available (e.g. `pusht_expert_train_long`)? Check the HuggingFace
   collection.
7. **Checkpoint organization.** Should Stage-2 checkpoints live
   alongside Stage-1 (`<run>/hlewm_epoch_*_object.ckpt`) or in a
   separate folder (`<run>/stage2/...`)? Plan recommends the latter to
   avoid `AutoCostModel` accidentally loading a Stage-2 file for the
   baseline.
8. **Outer CEM action space.** Faking a `gym.spaces.Box` to reuse
   `swm.solver.CEMSolver` vs writing a 60-line standalone CEM — which is
   cleaner? Plan prefers the standalone CEM for the outer level to
   avoid the `process['action']` coupling.
9. **VoE / surprise tests.** Out of scope for the first H-LeWM
   implementation? Plan defers.
10. **Naming convention.** Should the Stage-2 wrapper be `HLeWM`,
    `LeWMHierarchical`, or something else? Plan picks `HLeWM`.

---

## 10. Recommended Next Action

Run the **Step 1 smoke tests** locally on a small synthetic batch — no
training data required — to confirm that the LeWM module imports and
forward-passes correctly on the current machine. Specifically:

```python
# scratch.py (do not commit)
import torch
from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
# ... build a minimal JEPA with random encoder ...
# ... feed a (B=2, T=4, ...) batch and call lejepa_forward ...
```

Once that returns a finite loss, proceed to **Step 2** (encoder/predictor
API isolation) and **Step 4** (waypoint sampler) in parallel: they are
the smallest pieces of new code, fully decoupled from training and from
the planner, and each ships with a tight unit test that grounds the rest
of the project.

Only after Steps 1, 2, 4 are green should you start Stage-2 dataset
work (Step 5) and the two new modules (Steps 6, 7).
