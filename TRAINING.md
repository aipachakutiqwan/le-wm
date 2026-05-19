# Training LeWorldModel

## 1. Setup

### Docker workflow (recommended)

No venv needed. All dependencies run inside the container. Only requirements on the host:
- `python` — to run `devtools.py` (`fire` is auto-installed on first run)
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

## 4. Docker

All Docker operations go through `./devtools.py`. Image name is fixed as `cs231n_project/lewm`.

### Image tagging

When no `--tag` is provided, the tag is auto-generated from git state:

```
YYYYMMDD_GITHASH_GITBRANCH   e.g. 20260517_924f3ad_main
```

### Build

```bash
# auto-tagged from git
./devtools.py build_docker

# explicit tag
./devtools.py build_docker --tag test

# build and push to GHCR in one shot (requires GITHUB_PAT and GITHUB_USERNAME)
./devtools.py build_docker --push
./devtools.py build_docker --tag test --push
```

### Develop without rebuilding

Mounts the local repo at `/app` — edits on the host are instantly reflected inside the container:

```bash
./devtools.py dev test
```

Inside the container:

```bash
cd /app
python3 train.py data=pusht setup=local_rtx2080
```

### Test with baked image

`run_local` requires tag explicitly — no default:

```bash
./devtools.py run_local test
./devtools.py run_local test --data pusht
./devtools.py run_local test --data pusht --setup local_rtx2080

# pass extra Hydra overrides
./devtools.py run_local test --data pusht --overrides "[trainer.limit_train_batches=10,trainer.max_epochs=1,subdir=test]"
```

### Push to GHCR

**Generate a GitHub PAT (one-time):**
1. Go to https://github.com/settings/tokens → **Generate new token (classic)**
2. Select scopes: `write:packages`, `read:packages`
3. Copy the token

```bash
./devtools.py login
./devtools.py push_docker <tag>
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
./devtools.py login
./devtools.py pull_docker <tag>
```

The image is pulled from `ghcr.io/<image-owner-username>/lewm:<tag>` and automatically retagged locally as `cs231n_project/lewm:<tag>` — ready to use with `run_local` and `dev` immediately.

---

## 5. devtools.py reference

| Command | Description |
|---|---|
| `build_docker [--tag] [--push]` | Build image, optionally push to GHCR |
| `login` | Log in to GHCR using `GITHUB_PAT` and `GITHUB_USERNAME` env vars |
| `push_docker <tag>` | Tag and push image to GHCR |
| `pull_docker <tag>` | Pull image from GHCR and retag locally |
| `run_local <tag> [--data] [--setup] [--overrides]` | Run training in baked image |
| `dev <tag>` | Interactive shell with live repo mount |
