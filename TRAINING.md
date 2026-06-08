# Training LeWorldModel

## 1. Setup

### Docker workflow (recommended)

No venv needed. All dependencies run inside the container. Only requirements on the host:
- `python` — to run `scripts/devtools.py` (`fire` is auto-installed on first run)
- `docker` — to build and run containers
- `git` — for auto-tagging

```bash
export STABLEWM_HOME=<PATH_TO_stablewm-home>
export GITHUB_USERNAME=<your-github-username>
export GITHUB_PAT=<your-pat>
export WANDB_API_KEY=<your-key>   # optional, for W&B logging
```

### Local workflow (optional)

Only needed if running `train.py` directly outside Docker:

```bash
cd <PATH_TO_le-wm_repo>
uv sync --python <PATH_TO_PYTHON>
source .venv/bin/activate
export STABLEWM_HOME=<PATH_TO_stablewm-home>
```

---

## 2. Config

`lewm.yaml` defines all base values. `setup` configs patch only what differs per environment.
Priority: **CLI overrides > setup yaml > lewm.yaml**.

| Setup | File | GPU | batch | precision | wandb |
|---|---|---|---|---|---|
| _(none)_ | `lewm.yaml` | any | 128 | bf16 | on |
| `setup=local_rtx2080` | `config/train/setup/local_rtx2080.yaml` | RTX 2080 Ti (11 GB) | 64 | 16-mixed | on |
| `setup=cloud_a10g` | `config/train/setup/cloud_a10g.yaml` | A10G (24 GB) | 256 | bf16 | on |

### Dataset configs

| Arg | Dataset | File | Size |
|---|---|---|---|
| `data=pusht` | PushT | `pusht_expert_train.h5` | 44 GB |
| `data=tworoom` | TwoRoom | `tworoom.h5` | 12 GB |
| `data=dmc` | Reacher | `reacher.h5` | 93 GB |
| `data=ogb` | Cube | `cube_single_expert.h5` | 95 GB |

---

## 3. Train

```bash
# base defaults
python train.py data=pusht
python train.py data=tworoom

# with setup
python train.py data=pusht setup=local_rtx2080
python train.py data=pusht setup=cloud_a10g

# single param override on top of any setup
python train.py setup=local_rtx2080 data=pusht trainer.max_epochs=50
python train.py setup=cloud_a10g data=tworoom loader.batch_size=512 wandb.enabled=False
```

Checkpoints are saved to `$STABLEWM_HOME/<run_id>/` after each epoch:

| File | Purpose |
|---|---|
| `lewm_epoch_N_object.ckpt` | Model weights pickle per epoch (inference) |
| `lewm_weights.ckpt` | Full training state for resuming (overwritten each epoch) |

The resolved config is printed at startup — check `trainer.precision` and `loader.batch_size` to confirm the right setup is active.

### Quick testing

`limit_train_batches` behaves differently by type — **int = number of batches, float = fraction**:

```bash
# 10 batches per epoch (int)
python train.py trainer.limit_train_batches=10 trainer.limit_val_batches=5

# 1 batch per epoch (int) — minimal smoke test
python train.py trainer.fast_dev_run=True

# use a fresh subdir to avoid checkpoint epoch mismatch when testing
python train.py subdir=test trainer.limit_train_batches=10 trainer.max_epochs=1
```

> `limit_train_batches=1` (int) = 1 batch. `limit_train_batches=1.0` (float) = 100% of batches.
> Reference: https://lightning.ai/docs/pytorch/stable/common/trainer.html#limit-train-batches

### W&B checkpoint artifacts

Set `wandb.config.log_model` to upload checkpoints as named artifacts:

```bash
# upload best checkpoint only (artifact name includes epoch)
python train.py wandb.config.log_model=True

# upload all epoch checkpoints
python train.py wandb.config.log_model=all
```

Find artifacts at: `https://wandb.ai/<entity>/<project>/artifacts/model`

---

## 4. Eval

Evaluation runs live MuJoCo environments — requires a built checkpoint and the `tworoom` (or other) dataset in `$STABLEWM_HOME`.

Checkpoints tracked by Git LFS must be pulled on the host before eval — the container mounts `/app` from the host, so the real binary becomes visible immediately without a rebuild:

```bash
# on the host, before entering the container
git lfs pull --include="baseline/tworoom/lewm_epoch_9_object.ckpt"
```

Pass the **directory** containing the checkpoint as `policy`. `AutoCostModel` will pick the latest `*_object.ckpt` inside it automatically.

```bash
# inside the Docker dev shell (./scripts/devtools.py dev <tag>)
cd /app
python eval.py --config-name tworoom policy=/app/baseline/tworoom

# override eval episodes or planning horizon
python eval.py --config-name tworoom \
  policy=/app/baseline/tworoom \
  eval.num_eval=50 \
  plan_config.horizon=5

# custom output file
python eval.py --config-name tworoom \
  policy=/app/baseline/tworoom \
  output.filename=my_results.txt
```

> **Do not pass the full `.ckpt` filename** — `AutoCostModel` appends `_object.ckpt` itself and will double-suffix it.

Results are written to `Path($STABLEWM_HOME, policy).parent / output.filename`.
For `policy=/app/baseline/tworoom`, that resolves to `/app/baseline/tworoom_results.txt`.

### Eval config reference (`config/eval/tworoom.yaml`)

| Key | Default | Notes |
|---|---|---|
| `policy` | `random` | Directory or stem path to checkpoint |
| `eval.num_eval` | `50` | Number of episodes to evaluate |
| `eval.eval_budget` | `50` | Max steps per episode |
| `plan_config.horizon` | `5` | Planning horizon |
| `plan_config.action_block` | `5` | Actions executed per planning step |
| `output.filename` | `tworoom_results.txt` | Output file (appended, not overwritten) |

---

## 5. Docker

All Docker operations go through `./scripts/devtools.py`. Image name is fixed as `cs231n_project/lewm`.

> **Eval requires EGL.** The image installs `libegl1`, `libgl1`, and `libglfw3` so MuJoCo can render headlessly. `eval.py` sets `MUJOCO_GL=egl` automatically — no extra flags needed at runtime.

### Image tagging

When no `--tag` is provided, the tag is auto-generated from git state:

```
YYYYMMDD_GITHASH_GITBRANCH   e.g. 20260517_924f3ad_main
```

### Build

Builds target **`linux/amd64`** (CUDA / `box2d` wheels). On Apple Silicon, Docker Desktop would otherwise build `arm64` and fail at `uv sync` with a missing `box2d` wheel.

```bash
# auto-tagged from git
./scripts/devtools.py build_docker

# explicit tag
./scripts/devtools.py build_docker --tag test

# build and push to GHCR in one shot (requires GITHUB_PAT and GITHUB_USERNAME)
./scripts/devtools.py build_docker --push
./scripts/devtools.py build_docker --tag test --push
```

### Develop without rebuilding

Mounts the local repo at `/app` — edits on the host are instantly reflected inside the container:

```bash
./scripts/devtools.py dev test
```

Inside the container:

```bash
cd /app
python3 train.py data=pusht setup=local_rtx2080
```

### Test with baked image

`run_local` requires tag explicitly — no default:

```bash
./scripts/devtools.py run_local test
./scripts/devtools.py run_local test --data pusht
./scripts/devtools.py run_local test --data pusht --setup local_rtx2080

# pass extra Hydra overrides
./scripts/devtools.py run_local test --data pusht --overrides "[trainer.limit_train_batches=10,trainer.max_epochs=1,subdir=test]"
```

### Push to GHCR

**Generate a GitHub PAT (one-time):**
1. Go to https://github.com/settings/tokens → **Generate new token (classic)**
2. Select scopes: `write:packages`, `read:packages`
3. Copy the token

```bash
./scripts/devtools.py login
./scripts/devtools.py push_docker <tag>
```

### Grant access to collaborators

1. Go to `github.com/<your-username>?tab=packages` → select `lewm` → **Package Settings → Manage Access**
2. Add collaborators by GitHub username
3. Set **Read** (pull only) or **Write** (pull + push)

### Pull as a collaborator

Generate a GitHub PAT at https://github.com/settings/tokens with `read:packages` scope, then:

```bash
export GITHUB_USERNAME=<image-owner-username>
export GITHUB_PAT=<your-pat>
./scripts/devtools.py login
./scripts/devtools.py pull_docker <tag>
```

The image is pulled from `ghcr.io/<image-owner-username>/lewm:<tag>` and automatically retagged locally as `cs231n_project/lewm:<tag>` — ready to use with `run_local` and `dev` immediately.

---

## 6. devtools.py reference

| Command | Description |
|---|---|
| `build_docker [--tag] [--push]` | Build image, optionally push to GHCR |
| `login` | Log in to GHCR using `GITHUB_PAT` and `GITHUB_USERNAME` env vars |
| `push_docker <tag>` | Tag and push image to GHCR |
| `pull_docker <tag>` | Pull image from GHCR and retag locally |
| `run_local <tag> [--data] [--setup] [--overrides]` | Run stage-1 training in baked image |
| `run_hierarchical_local <tag> --stage1-checkpoint <path> [--data] [--setup] [--overrides] [--dry-run]` | Run stage-2 hierarchical training in baked image |
| `run_hierarchical_modal <tag> --stage1-checkpoint <path> [--data] [--overrides] [--dry-run]` | Submit stage-2 hierarchical training to Modal (A100) |
| `dev <tag>` | Interactive shell with live repo mount |


## 7. Hierarchical LeWM (stage 2)

Stage-2 training freezes the stage-1 JEPA and jointly trains the `ActionEncoder` (A_ψ)
and `HighLevelPredictor` (P^(2)) on a teacher-forcing waypoint loss.

### Prerequisites

You need a stage-1 checkpoint (`lewm_epoch_N_object.ckpt`) from stage-1 training or the
paper weights (see `scripts/convert_paper_weights.py`).

---

### Local (direct, no Docker) — quick smoke-test

```bash
# CPU smoke-test (2 epochs, tiny batch)
python train_hierarchical.py \
  data=tworoom \
  stage1_checkpoint=<path/to/lewm_epoch_100_object.ckpt> \
  setup=cpu \
  stage2.n_epochs=2 \
  loader.batch_size=8 \
  wandb.enabled=False

# GPU — full run
python train_hierarchical.py \
  data=tworoom \
  stage1_checkpoint=<path/to/lewm_epoch_100_object.ckpt> \
  stage2.n_epochs=50
```

The hierarchical model is saved to `$STABLEWM_HOME/<subdir>/hierarchical_lewm_object.ckpt`.

---

### Local (Docker container)

The Docker image already contains `train_hierarchical.py`. Override the entrypoint at runtime:

```bash
# Dry-run (2 epochs, batch=8, no W&B)
./scripts/devtools.py run_hierarchical_local <tag> \
  --stage1-checkpoint /stablewm-home/lewm_epoch_100_object.ckpt \
  --dry-run

# Full run
./scripts/devtools.py run_hierarchical_local <tag> \
  --stage1-checkpoint /stablewm-home/lewm_epoch_100_object.ckpt \
  --data tworoom

# With extra Hydra overrides
./scripts/devtools.py run_hierarchical_local <tag> \
  --stage1-checkpoint /stablewm-home/lewm_epoch_100_object.ckpt \
  --overrides "[stage2.n_epochs=50,wandb.enabled=False]"
```

The checkpoint path is relative to the container. Files in `STABLEWM_HOME` are at
`/stablewm-home/...`; Git-tracked files are at `/app/baseline/...`.

---

### Modal (A100 GPU)

#### 1. Upload the stage-1 checkpoint to the Modal volume (if not already there)

```bash
modal volume put lewm-data \
  <local_path>/lewm_epoch_100_object.ckpt \
  lewm_epoch_100_object.ckpt
```

The file will be at `/stablewm-home/lewm_epoch_100_object.ckpt` inside the container.

#### 2. Submit the job

```bash
# Dry-run (~5 min, 2 epochs, A100)
./scripts/devtools.py run_hierarchical_modal <tag> \
  --stage1-checkpoint /stablewm-home/lewm_epoch_100_object.ckpt \
  --dry-run

# Full run
./scripts/devtools.py run_hierarchical_modal <tag> \
  --stage1-checkpoint /stablewm-home/lewm_epoch_100_object.ckpt \
  --data tworoom

# Or directly via modal run
LEWM_TAG=<tag> modal run cloud/modal_train.py::train_hier \
  --stage1-checkpoint /stablewm-home/lewm_epoch_100_object.ckpt \
  --data tworoom \
  --overrides "stage2.n_epochs=50,wandb.enabled=True"
```

The trained model is committed to the Modal volume at
`/stablewm-home/<subdir>/hierarchical_lewm_object.ckpt`.

#### 3. Download the hierarchical checkpoint

```bash
modal volume get lewm-data <subdir>/hierarchical_lewm_object.ckpt ./
```

---

### Planning with the hierarchical model

```bash
python plan_hierarchical.py \
  checkpoint=<path/to/hierarchical_lewm_object.ckpt> \
  device=cpu \
  eval.num_eval=5 \
  plan.outer_samples=64 \
  plan.inner_samples=32
```

---

### Setup configs for stage-2

| Setup | File | GPU | batch |
|---|---|---|---|
| _(none)_ | `hierarchical.yaml` defaults | any | 64 |
| `setup=cloud_a10g` | `config/train/setup/cloud_a10g.yaml` | A10G (24 GB) | 192 |
| `setup=cloud_a100` | `config/train/setup/cloud_a100.yaml` | A100 (40/80 GB) | 256 |
| `setup=cpu` | `config/train/setup/cpu.yaml` | CPU | — |



### Hierarchical run:

Cube:

```bash
export STABLEWM_HOME="/var/cs231n/lewm-cube"
python train_hierarchical.py data=ogb stage1_checkpoint=$STABLEWM_HOME/lewm_paper_object.ckpt  wandb.config.entity=florenciopaucar-uni stage2.n_epochs=20 stage2.rollout_loss=True wm.lambda_kl=0.03 num_gpus=2 
```

```bash
export STABLEWM_HOME="/var/cs231n/lewm-cube"
python plan_hierarchical.py checkpoint=$STABLEWM_HOME/20260602_211303/hierarchical_lewm_best_object.ckpt device=cuda --config-name=hierarchical_cube

```