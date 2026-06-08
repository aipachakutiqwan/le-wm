# H-LeWM
### Hierarchical Planning with End-to-End JEPA World Models

Florencio Paucar, Adhavan Jayabalan, Arihant Jain (Stanford University)

> **This repository is a fork of [`lucas-maes/le-wm`](https://github.com/lucas-maes/le-wm)** (LeWorldModel).
> We forked the original LeWM codebase and did all of our development on top of it. The upstream
> code provides the stable end-to-end JEPA world model (encoder `E` + low-level predictor `P¹`);
> **our contribution, H-LeWM, adds a hierarchical macro-action layer on top of it** for long-horizon
> planning. See [Relationship to upstream LeWM](#relationship-to-upstream-lewm) for exactly what we added.

**Abstract:** Latent world models such as LeWorldModel (LeWM) suffer from compounding prediction
errors when unrolled over many steps, effectively restricting planning to short horizons. We propose
**H-LeWM**, an extension of LeWM to long-horizon control that adapts the Hierarchical Planning with
Latent World Models (HWM) framework to LeWM's stable, end-to-end JEPA backbone. H-LeWM leaves LeWM's
encoder and low-level predictor unchanged and trains, in a second stage, a learned **action encoder
`A_ψ`** that compresses chunks of primitive actions into compact latent **macro-actions**, together
with a **high-level predictor `P²`** that predicts waypoint-subsampled latent states conditioned on
those macro-actions. At inference, a **two-level CEM-MPC** first optimizes macro-actions at the high
level to produce intermediate subgoals, which the low-level predictor then pursues with primitive
actions. We evaluate across four environments (TwoRoom, PushT, OGBench-Cube, Reacher) and analyze
where the hierarchy helps — and where the underlying latent cost geometry, not the planner, is the
bottleneck.

---

## Method overview

H-LeWM is a two-stage model. Stage 1 is the original LeWM; Stage 2 adds the hierarchy.

| Component | Symbol | Trained in | Role |
|---|---|---|---|
| Encoder | `E` | Stage 1 (frozen in Stage 2) | maps an image `o_t` to a latent `z_t` |
| Low-level predictor | `P¹` | Stage 1 (frozen in Stage 2) | `ẑ_{t+1} = P¹(z_t, a_t)` from a primitive action |
| Action encoder | `A_ψ` | **Stage 2** | compresses an action chunk into a macro-action `l_k` |
| High-level predictor | `P²` | **Stage 2** | `ẑ_{t_{k+1}} = P²(z_{t_k}, l_k)` over waypoints |

**Stage 2 training.** `E` and `P¹` are frozen. `A_ψ` (a small `[CLS]`-token transformer, 8-d macro-actions)
and `P²` (same architecture as `P¹`, conditioned on a macro-action) are trained jointly via teacher
forcing on a waypoint-prediction loss, with an autoregressive rollout loss (scheduled sampling) and a
moment-matching penalty that drives the macro-action distribution toward `N(0, I)` so the CEM prior is
well-matched.

**Two-level CEM-MPC planning.** Given a start and goal image, an **outer CEM** searches over
macro-actions (rolling `P²` forward, scored by L1 distance to the encoded goal) to pick a **subgoal**;
an **inner CEM** then searches over primitive actions (rolling `P¹` forward) to reach that subgoal.
Only the first primitive action is executed before replanning (receding horizon).

A detailed, code-linked walkthrough of training and inference is in
[`plans/hlewm_train_test_methodology.md`](plans/hlewm_train_test_methodology.md).

---

## Relationship to upstream LeWM

This fork keeps the upstream LeWM training/eval stack (`train.py`, `eval.py`, `jepa.py`, `module.py`)
intact and **adds** the hierarchical layer and analysis tooling:

**New code (our contribution)**
- `hierarchical_lewm.py` — `HierarchicalLeWM`, `ActionEncoder` (`A_ψ`), `HighLevelPredictor` (`P²`), Stage-2 training
- `train_hierarchical.py` — Hydra driver for Stage-2 training
- `hierarchical_plan.py` — two-level CEM and the `plan()` routine
- `plan_hierarchical.py` — Hydra driver for hierarchical evaluation
- `waypoint_sampler.py` — waypoint subsampling for Stage-2 windows
- `config/train/hierarchical.yaml`, `config/train/setup/*` — Stage-2 configs
- `config/eval/hierarchical_{tworoom,pusht,cube,reacher}.yaml` — hierarchical eval configs
- `qualitative analysis/` — heat-map cost-landscape, latent-analysis macro-action probe, path-trajectory, and diagnostics scripts
- `results/`, `long_horizon_experiments/` — our evaluation outputs and long-horizon sweep
- `cloud/`, `devtools.py`, Docker workflow — multi-GPU and Modal/GCP training infrastructure

**Upstream LeWM (unchanged contribution of the original authors)**
- The end-to-end JEPA model and SIGReg objective (`jepa.py`, `module.py`, `train.py`, `eval.py`)
- This codebase builds on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for
  environment management, planning, and evaluation, and
  [stable-pretraining](https://github.com/galilai-group/stable-pretraining) for training.

For Stage 1 we directly use the pretrained LeWM weights released by the original authors on
[Hugging Face](https://huggingface.co/collections/quentinll/lewm) (converted via `convert_paper_weights.py`),
so no Stage-1 retraining is required to reproduce our results.

---

## Installation

```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

`STABLEWM_HOME` (defaults to `~/.stable-wm/`) is the dataset/checkpoint cache:

```bash
export STABLEWM_HOME=/path/to/your/storage
```

## Data

Datasets use HDF5 for fast loading. The four environments are TwoRoom, PushT, OGBench-Cube, and
Reacher. Download from [Hugging Face](https://huggingface.co/collections/quentinll/lewm) and decompress:

```bash
# git-lfs is required
git lfs install
git clone git@hf.co:datasets/quentinll/lewm-tworooms
cd lewm-tworooms && tar --zstd -xvf archive.tar.zst
```

Place the extracted `.h5` files under `$STABLEWM_HOME`. Dataset names are referenced without the
`.h5` extension (e.g. `config/train/data/pusht.yaml` resolves `pusht_expert_train` to
`$STABLEWM_HOME/pusht_expert_train.h5`).

---

## Stage 1 — base LeWM (upstream)

`jepa.py` contains the LeWM implementation; training is configured via Hydra under `config/train/`.

```bash
python train.py data=pusht
python train.py data=tworoom
```

Checkpoints are saved to `$STABLEWM_HOME`. To skip Stage 1, use the released paper weights:

```bash
python convert_paper_weights.py --only tworooms
```

See [`TRAINING.md`](TRAINING.md) for the full setup matrix, Docker workflow, and `devtools.py` reference.

## Stage 2 — hierarchical training (H-LeWM)

Stage 2 freezes the Stage-1 JEPA and jointly trains `A_ψ` and `P²`:

```bash
# quick CPU smoke-test
python train_hierarchical.py \
  data=tworoom \
  stage1_checkpoint=<path/to/lewm_epoch_N_object.ckpt> \
  setup=cpu stage2.n_epochs=2 loader.batch_size=8 wandb.enabled=False

# full GPU run (with rollout loss + moment-matching reg)
python train_hierarchical.py \
  data=tworoom \
  stage1_checkpoint=<path/to/lewm_epoch_N_object.ckpt> \
  stage2.n_epochs=20 stage2.rollout_loss=True wm.lambda_kl=0.03
```

The hierarchical model is saved to `$STABLEWM_HOME/<subdir>/hierarchical_lewm_object.ckpt`.
Modal (A100) and Docker workflows are documented in [`TRAINING.md`](TRAINING.md) §7.

## Planning / evaluation

Flat LeWM planning (upstream):

```bash
python eval.py --config-name=tworoom.yaml policy=tworoom/lewm
```

Hierarchical H-LeWM planning (two-level CEM-MPC):

```bash
python plan_hierarchical.py \
  --config-name=hierarchical_tworoom \
  checkpoint=<path/to/hierarchical_lewm_object.ckpt> \
  device=cuda \
  eval.num_eval=50 eval.goal_offset_steps=25 eval.eval_budget=50 \
  plan.h_low=3 plan.outer_std=2.5
```

Hierarchical eval configs are provided for all four environments:
`hierarchical_tworoom`, `hierarchical_pusht`, `hierarchical_cube`, `hierarchical_reacher`.

---

## Results

**Long-horizon success rate (%)** vs. goal offset `Δ` (eval budget `= 2Δ`, `N=50`, same frozen
encoder), flat LeWM vs. hierarchical H-LeWM:

| Env | Planner | Δ=25 | Δ=50 | Δ=75 | Δ=100 |
|---|---|:---:|:---:|:---:|:---:|
| TwoRoom | Flat LeWM | 88 | 50 | 34 | 12 |
| TwoRoom | H-LeWM | 72 | 32 | 18 | 10 |
| Reacher | Flat LeWM | 78 | 94 | 88 | 80 |
| Reacher | H-LeWM | 62 | 72 | 78 | 68 |

**Key finding.** Across every measured horizon the flat planner does at least as well as the
hierarchical one. Our analysis shows the bottleneck is **the latent cost geometry, not the planner**:
`‖E(s) − E(g)‖` saturates beyond a short local basin (~30 arena units on TwoRoom), so the macro
planner cannot rank far-away subgoals. TwoRoom collapses as `Δ` grows because its goals move into that
"dead zone"; Reacher stays flat only because its goals remain physically close. The macro-action
encoder itself is well-behaved — a linear probe recovers per-chunk net motion at CV `R²=0.89`
(`R²_Δy=0.98`, `R²_Δx=0.80`), confirming the failure is upstream of the hierarchy.

The TwoRoom tuning progression that took Stage-2 success from 20% → 82% (at `Δ=25`) is documented in
[`RUNS.md`](RUNS.md) and the paper's ablation table.

## Qualitative analysis

Offline, environment-free diagnostics (encoder/predictor forward passes only) live under
`qualitative analysis/`:

| Folder | Produces |
|---|---|
| `heat maps/` | latent cost-landscape over true `(x,y)`; cost-vs-distance saturation curve |
| `latent_analysis/` | macro-action linear-probe figure (`A_ψ` codes → net motion) |
| `path_trajectories/` | flat vs. hierarchical rollout trajectory overlays |
| `diagnostics/` | `hierarchical_probe.py` — decoupled `A_ψ`/`P²`/`P¹` fidelity checks |

Each folder has its own README. Quote the script path when running (the folder name contains a space):

```bash
STABLEWM_HOME=$HOME/.stable_worldmodel \
  .venv/bin/python "qualitative analysis/heat maps/cost_landscape.py" --device cuda
```

---

## Citation

This work extends LeWorldModel. If you use this code, please cite the original LeWM paper:

```bibtex
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

LeWM resources: [Paper](https://arxiv.org/pdf/2603.19312v1) ·
[Checkpoints](https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e) ·
[Data](https://huggingface.co/collections/quentinll/lewm) ·
[Website](https://le-wm.github.io/) ·
[Upstream repo](https://github.com/lucas-maes/le-wm)

## Acknowledgements

H-LeWM is built directly on top of [`lucas-maes/le-wm`](https://github.com/lucas-maes/le-wm) and the
[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) /
[stable-pretraining](https://github.com/galilai-group/stable-pretraining) libraries. We thank the LeWM
authors for releasing their code, pretrained checkpoints, and datasets.
