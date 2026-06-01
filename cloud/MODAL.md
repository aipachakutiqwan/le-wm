# Modal Training Setup

Training on Modal using the pre-built GHCR image. Modal auto-terminates
the instance when training finishes — no manual shutdown needed.

---

## Prerequisites

```bash
pip install modal
modal setup   # opens browser to link your Modal account
```

---

## One-Time Setup

These steps create the persistent resources that are reused across all runs.

### 1. Create the dataset volume

```bash
modal volume create lewm-data
```

### 2. Download datasets into the volume

Download directly from HuggingFace into the Modal volume — much faster than
uploading from local (Modal datacenter pulls at ~500 MB/s vs your local upload speed).

```bash
# tworoom only (12 GB, ~1 min)
./devtools.py download_modal tworoom

# multiple datasets
./devtools.py download_modal tworoom pusht

# or via modal run directly (no tag needed — uses lightweight image)
modal run cloud/modal_train.py::download --envs tworoom
```

Available envs: `tworoom` (12 GB), `pusht` (44 GB), `reacher` (93 GB), `cube` (95 GB)

To verify what's on the volume:
```bash
modal volume ls lewm-data /
```

### 3. Create secrets

```bash
# W&B logging
modal secret create wandb-secret \
  WANDB_API_KEY=<your-wandb-api-key>

# GHCR credentials to pull the private training image
modal secret create ghcr-secret \
  REGISTRY_USERNAME=<your-github-username> \
  REGISTRY_PASSWORD=<your-github-pat>
```

To verify secrets exist:
```bash
modal secret list
```

---

## Running Training

### Via devtools.py (recommended)

```bash
# Dry-run — 10 batches, 1 epoch, no W&B (~2 min, quick sanity check)
./devtools.py run_modal <tag> --dry-run

# Full training run
./devtools.py run_modal <tag> \
  --overrides="[loader.batch_size=192,loader.prefetch_factor=2,wandb.enabled=True,wandb.config.entity='addyj-stanford-university',wandb.config.project='le-wm-test']"

# Different dataset
./devtools.py run_modal <tag> --data pusht
```

### Via modal run directly

The image tag is passed via the `LEWM_TAG` environment variable (Modal evaluates
decorators at import time, before CLI args are parsed):

```bash
# Dry-run
LEWM_TAG=<tag> modal run cloud/modal_train.py --dry-run

# Full run with overrides
LEWM_TAG=<tag> modal run cloud/modal_train.py \
  --data tworoom \
  --setup cloud_a10g \
  --overrides "loader.batch_size=192,loader.prefetch_factor=2,wandb.enabled=True"
```

`devtools.py run_modal` sets `LEWM_TAG` automatically — prefer that for day-to-day use.

Get the current tag with:
```bash
./devtools.py _git_tag
```

---

## Running Eval

Eval runs on an **A10G** (cheaper than A100 — planning is inference-only). The checkpoint must be in the Modal volume at `/stablewm-home`.

### Upload a local checkpoint to the volume

**For your own trained checkpoints** (already in `_object.ckpt` format):
```bash
modal volume put lewm-data baseline/tworoom/lewm_epoch_9_object.ckpt lewm_epoch_9_object.ckpt
```

**For paper weights** (`weights.pt` from HuggingFace), convert first:
```bash
# converts weights.pt + config.json → baseline_paper/tworooms/lewm_paper_object.ckpt
python convert_paper_weights.py --only tworooms

# then upload the converted checkpoint
modal volume put lewm-data baseline_paper/tworooms/lewm_paper_object.ckpt lewm_paper_object.ckpt
```

After upload, the checkpoint is at `/stablewm-home/lewm_paper_object.ckpt` inside the container.

### Via devtools.py (recommended)

```bash
# evaluate the paper weights (after convert + upload above)
./devtools.py eval_modal <tag> --policy /stablewm-home/lewm_paper

# evaluate a trained checkpoint
./devtools.py eval_modal <tag> --policy /stablewm-home/lewm_epoch_9

# evaluate a checkpoint from a training run
./devtools.py eval_modal <tag> --policy /stablewm-home/<run_id>

# with overrides
./devtools.py eval_modal <tag> \
  --policy /stablewm-home/lewm_epoch_9 \
  --overrides "eval.num_eval=10,plan_config.horizon=5"
```

### Via modal run directly

```bash
LEWM_TAG=<tag> modal run cloud/modal_train.py::eval \
  --policy /stablewm-home/lewm_epoch_9

LEWM_TAG=<tag> modal run cloud/modal_train.py::eval \
  --policy /stablewm-home/lewm_epoch_9 \
  --overrides "eval.num_eval=10,plan_config.horizon=5"
```

> **Policy path convention:** pass the directory or the path stem *without* `_object.ckpt`.
> `AutoCostModel` appends `_object.ckpt` itself — passing the full filename will double-suffix it.

Results are written to the volume at `Path(policy).parent / tworoom_results.txt` and persisted via `volume.commit()`.

---

## Monitoring

Stream logs live while the job runs:
```bash
modal app logs lewm-training
```

Check running jobs:
```bash
modal app list
```

Stop a running job:
```bash
modal app stop lewm-training
```

---

## Costs

- **A10G GPU**: ~$1.10/hr on Modal
- **Volume storage**: ~$0.15/GB/month (lewm-data ~165 GB full, ~12 GB tworoom only)
- **Egress**: free within Modal

Estimated cost for a full 100-epoch tworoom run (~3–4 hrs): **~$4–5**

---

## Rebuilding the Image

If you've changed the Dockerfile or dependencies, rebuild and push to GHCR first,
then pass the new tag to `run_modal`:

```bash
./devtools.py build_docker --push
./devtools.py run_modal <new-tag>
```

---

## devtools.py reference

| Command | Description |
|---|---|
| `download_modal <env>` | Download dataset into Modal volume from HuggingFace |
| `run_modal <tag> [--data] [--setup] [--overrides] [--dry-run]` | Submit training job (A100) |
| `eval_modal <tag> --policy <path> [--config] [--overrides]` | Submit eval job (A10G) |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No such secret: ghcr-secret` | Secret not created | Run `modal secret create ghcr-secret ...` |
| `No such volume: lewm-data` | Volume not created | Run `modal volume create lewm-data` |
| `pull access denied` for GHCR image | Wrong PAT or username in ghcr-secret | Recreate secret with correct credentials |
| `FileNotFoundError: tworoom.h5` | Dataset not uploaded to volume | Run `modal volume put lewm-data ...` |
| `UnpicklingError: invalid load key, 'v'` | LFS pointer instead of real checkpoint | Run `git lfs pull` on host, or `modal volume put` the real binary |
| OOM during training | batch_size too large | Add `loader.batch_size=128` to overrides |
| Job times out | 16h limit exceeded | Reduce epochs or increase `timeout` in `modal_train.py` |
