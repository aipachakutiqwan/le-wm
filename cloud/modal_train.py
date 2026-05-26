"""Modal training job for LeWM.

Uses the pre-built GHCR image so the Modal environment exactly matches
the GCP/local dev setup — no Dockerfile rebuild needed.

Modal evaluates @app.function decorators at import time on your local machine,
so the image tag is passed via the LEWM_TAG environment variable rather than
as a CLI argument. devtools.py sets this automatically.

One-time setup (run once per machine):
    pip install modal
    modal setup

    modal volume create lewm-data
    modal secret create wandb-secret WANDB_API_KEY=<key>
    modal secret create ghcr-secret REGISTRY_USERNAME=<user> REGISTRY_PASSWORD=<pat>

Usage:
    # Download dataset into Modal volume (no tag needed)
    ./devtools.py download_modal tworoom

    # Dry-run training (quick sanity check)
    ./devtools.py run_modal <tag> --dry-run

    # Full training run
    ./devtools.py run_modal <tag>

    # Or directly via modal (tag passed via env var)
    LEWM_TAG=<tag> modal run cloud/modal_train.py --data tworoom
    LEWM_TAG=<tag> modal run cloud/modal_train.py --dry-run
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GHCR_OWNER = "jadhavan"
IMAGE_NAME  = "lewm"
GPU         = "A100"
EVAL_GPU    = "A10G"
_REPO_ROOT  = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def make_image(tag: str) -> modal.Image:
    # psutil is used by the resource monitor thread below; it's a transitive of
    # ipykernel (a dev-only dep) so `uv sync --no-dev` in the Dockerfile drops it.
    # The image's venv at /opt/venv has no pip (uv-managed), so use uv from its
    # install location to add psutil into that venv.
    return modal.Image.from_registry(
        f"ghcr.io/{GHCR_OWNER}/{IMAGE_NAME}:{tag}",
        secret=modal.Secret.from_name("ghcr-secret"),
    ).run_commands(
        "/root/.local/bin/uv pip install --python /opt/venv/bin/python psutil"
    )


# Lightweight image for dataset downloads — embeds download_datasets.py so
# the full GHCR image is not needed just to pull data from HuggingFace.
DOWNLOAD_IMAGE = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub", "zstandard")
    .add_local_file(str(_REPO_ROOT / "download_datasets.py"), "/app/download_datasets.py")
)

# Training image — resolved from LEWM_TAG env var at import time.
# devtools.py sets this automatically; for direct modal run use:
#   LEWM_TAG=<tag> modal run cloud/modal_train.py
_tag = os.environ.get("LEWM_TAG")
if not _tag:
    # No tag set — use download image as a harmless placeholder so the module
    # loads cleanly when only the download entrypoint is being used.
    TRAIN_IMAGE = DOWNLOAD_IMAGE
else:
    TRAIN_IMAGE = make_image(_tag)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = modal.App("lewm-training")

volume = modal.Volume.from_name("lewm-data", create_if_missing=False)

# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------


@app.function(
    image=DOWNLOAD_IMAGE,
    volumes={"/stablewm-home": volume},
    timeout=60 * 60 * 4,
    cpu=4,
)
def download_data(envs: list[str] = ("tworoom",)) -> None:
    """Download and extract LeWM datasets from HuggingFace into the Modal volume."""
    os.environ["STABLEWM_HOME"] = "/stablewm-home"

    result = subprocess.run(
        [sys.executable, "/app/download_datasets.py", "--only", *envs],
        cwd="/app",
        check=False,
    )

    volume.commit()

    if result.returncode != 0:
        raise RuntimeError(f"download_datasets.py failed with exit code {result.returncode}")

    print("[download] done — datasets ready in /stablewm-home", flush=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@app.function(
    image=TRAIN_IMAGE,
    gpu=GPU,
    volumes={"/stablewm-home": volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=60 * 60 * 16,
    retries=0,
)
def train(
    data: str = "tworoom",
    setup: str = "cloud_a10g",
    overrides: Optional[list[str]] = None,
    subdir: Optional[str] = None,
    monitor_interval: int = 60,
) -> int:
    """Run train.py inside the container with Hydra overrides."""
    import threading
    import psutil

    stop_event = threading.Event()

    def _monitor():
        gpu_query = "utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"
        header = f"{'GPU%':>5} {'GMEM%':>6} {'VRAM':>12}  {'CPU%':>5} {'RAM':>12}"
        print(f"[res] {header}", flush=True)
        while not stop_event.wait(timeout=monitor_interval):
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", f"--query-gpu={gpu_query}", "--format=csv,noheader,nounits"],
                    text=True,
                ).strip()
                gpu_util, gmem_util, vram_used, vram_total, power = [x.strip() for x in out.split(",")]
                vram_str = f"{vram_used}/{vram_total}MiB"
            except Exception:
                gpu_util = gmem_util = vram_str = power = "n/a"

            cpu_pct = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory()
            ram_str = f"{ram.used/1e9:.1f}/{ram.total/1e9:.1f}GB"
            print(
                f"[res] {gpu_util:>4}%  {gmem_util:>5}%  {vram_str:>12}  {cpu_pct:>4}%  {ram_str:>12}",
                flush=True,
            )

    threading.Thread(target=_monitor, daemon=True).start()

    cmd = ["python", "train.py", f"data={data}", f"setup={setup}"]
    if subdir:
        cmd += [f"subdir={subdir}"]
    if overrides:
        cmd += overrides

    print(f"[modal] running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd="/app", check=False)

    stop_event.set()

    if result.returncode != 0:
        print(f"[modal] training failed with exit code {result.returncode}", flush=True)
        sys.exit(result.returncode)

    return result.returncode


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


@app.function(
    image=TRAIN_IMAGE,
    gpu=EVAL_GPU,
    volumes={"/stablewm-home": volume},
    timeout=60 * 60 * 2,
    retries=0,
)
def eval_job(
    policy: str,
    config: str = "tworoom",
    overrides: Optional[list[str]] = None,
) -> None:
    """Run eval.py inside the container with Hydra overrides.

    Args:
        policy:    Path to the checkpoint directory or stem inside /stablewm-home.
                   e.g. "/stablewm-home/lewm_epoch_9" or "/stablewm-home/<run_id>"
        config:    Hydra config name under config/eval/ (default: tworoom)
        overrides: Extra Hydra overrides, e.g. ["eval.num_eval=10"]
    """
    cmd = [
        "python", "eval.py",
        f"--config-name={config}",
        f"policy={policy}",
    ]
    if overrides:
        cmd += overrides

    print(f"[modal] running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd="/app", check=False)

    volume.commit()

    if result.returncode != 0:
        print(f"[modal] eval failed with exit code {result.returncode}", flush=True)
        sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Hierarchical training (stage 2)
# ---------------------------------------------------------------------------


@app.function(
    image=TRAIN_IMAGE,
    gpu=GPU,
    volumes={"/stablewm-home": volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=60 * 60 * 16,
    retries=0,
)
def train_hierarchical(
    stage1_checkpoint: str,
    data: str = "tworoom",
    overrides: Optional[list[str]] = None,
    subdir: Optional[str] = None,
    monitor_interval: int = 60,
) -> int:
    """Run train_hierarchical.py (stage-2) inside the container with Hydra overrides.

    Args:
        stage1_checkpoint: Absolute path to the stage-1 .ckpt inside /stablewm-home.
                           e.g. "/stablewm-home/lewm_epoch_100_object.ckpt"
        data:              Hydra data config (tworoom, pusht, reacher, cube)
        overrides:         List of Hydra overrides, e.g. ["stage2.n_epochs=50"]
        subdir:            Optional output subdirectory name inside STABLEWM_HOME.
        monitor_interval:  Seconds between resource log lines.
    """
    import threading
    import psutil

    stop_event = threading.Event()

    def _monitor():
        gpu_query = "utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"
        header = f"{'GPU%':>5} {'GMEM%':>6} {'VRAM':>12}  {'CPU%':>5} {'RAM':>12}"
        print(f"[res] {header}", flush=True)
        while not stop_event.wait(timeout=monitor_interval):
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", f"--query-gpu={gpu_query}", "--format=csv,noheader,nounits"],
                    text=True,
                ).strip()
                gpu_util, gmem_util, vram_used, vram_total, power = [x.strip() for x in out.split(",")]
                vram_str = f"{vram_used}/{vram_total}MiB"
            except Exception:
                gpu_util = gmem_util = vram_str = power = "n/a"

            cpu_pct = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory()
            ram_str = f"{ram.used/1e9:.1f}/{ram.total/1e9:.1f}GB"
            print(
                f"[res] {gpu_util:>4}%  {gmem_util:>5}%  {vram_str:>12}  {cpu_pct:>4}%  {ram_str:>12}",
                flush=True,
            )

    threading.Thread(target=_monitor, daemon=True).start()

    cmd = [
        "python", "train_hierarchical.py",
        f"stage1_checkpoint={stage1_checkpoint}",
        f"data={data}",
    ]
    if subdir:
        cmd += [f"subdir={subdir}"]
    if overrides:
        cmd += overrides

    print(f"[modal] running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd="/app", check=False)

    stop_event.set()
    volume.commit()  # persist the trained hierarchical model to the volume

    if result.returncode != 0:
        print(f"[modal] hierarchical training failed with exit code {result.returncode}", flush=True)
        sys.exit(result.returncode)

    return result.returncode


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def download(envs: str = "tworoom"):
    """Download datasets directly into the Modal volume from HuggingFace.

    Args:
        envs: Comma-separated env names. Choices: tworoom, pusht, cube, reacher
              e.g. --envs tworoom  or  --envs tworoom,pusht
    """
    env_list = [e.strip() for e in envs.split(",") if e.strip()]
    print(f"[local] downloading datasets: {env_list}")
    download_data.remote(envs=env_list)


@app.local_entrypoint()
def main(
    data: str = "tworoom",
    setup: str = "cloud_a10g",
    overrides: str = "",
    dry_run: bool = False,
):
    """Submit a training job. Set LEWM_TAG env var to select the image.

    Args:
        data:      Hydra data config (tworoom, pusht, reacher, cube)
        setup:     Hydra setup config (cloud_a10g)
        overrides: Comma-separated Hydra overrides
                   e.g. "trainer.max_epochs=5,wandb.enabled=False"
        dry_run:   Limits to 10 batches / 1 epoch, disables W&B
    """
    if not _tag:
        raise SystemExit("LEWM_TAG env var is not set. Use ./devtools.py run_modal <tag> or: LEWM_TAG=<tag> modal run cloud/modal_train.py")

    override_list = [o.strip() for o in overrides.split(",") if o.strip()]

    if dry_run:
        override_list += [
            "trainer.limit_train_batches=10",
            "trainer.max_epochs=1",
            "trainer.limit_val_batches=5",
            "wandb.enabled=False",
        ]

    print(f"[local] submitting job — image: {_tag}, data: {data}, setup: {setup}")
    print(f"[local] overrides: {override_list}")

    train.remote(data=data, setup=setup, overrides=override_list)


@app.local_entrypoint()
def eval(
    policy: str,
    config: str = "tworoom",
    overrides: str = "",
):
    """Submit an eval job to Modal. Set LEWM_TAG env var to select the image.

    Args:
        policy:    Path to checkpoint directory or stem inside /stablewm-home.
                   e.g. "/stablewm-home/lewm_epoch_9" or "/stablewm-home/<run_id>"
        config:    Hydra eval config name (default: tworoom)
        overrides: Comma-separated Hydra overrides
                   e.g. "eval.num_eval=10,plan_config.horizon=5"
    """
    if not _tag:
        raise SystemExit("LEWM_TAG env var is not set. Use ./devtools.py eval_modal <tag> or: LEWM_TAG=<tag> modal run cloud/modal_train.py::eval --policy <policy>")

    override_list = [o.strip() for o in overrides.split(",") if o.strip()]

    print(f"[local] submitting eval — image: {_tag}, config: {config}, policy: {policy}")
    print(f"[local] overrides: {override_list}")

    eval_job.remote(policy=policy, config=config, overrides=override_list)


@app.local_entrypoint()
def train_hier(
    stage1_checkpoint: str,
    data: str = "tworoom",
    overrides: str = "",
    dry_run: bool = False,
):
    """Submit a stage-2 hierarchical training job (A100). Set LEWM_TAG env var to select the image.

    The stage-1 checkpoint must already be in the Modal volume at /stablewm-home.
    Upload it first if needed:
        modal volume put lewm-data <local_path>_object.ckpt <filename>_object.ckpt

    Args:
        stage1_checkpoint: Absolute path inside /stablewm-home.
                           e.g. "/stablewm-home/lewm_epoch_100_object.ckpt"
        data:      Hydra data config (tworoom, pusht, reacher, cube)
        overrides: Comma-separated Hydra overrides
                   e.g. "stage2.n_epochs=50,loader.batch_size=128,wandb.enabled=True"
        dry_run:   Limits to 2 epochs, batch_size=8, disables W&B (~5 min sanity check)
    """
    if not _tag:
        raise SystemExit(
            "LEWM_TAG env var is not set. "
            "Use ./devtools.py run_hierarchical_modal <tag> --stage1-checkpoint <path> "
            "or: LEWM_TAG=<tag> modal run cloud/modal_train.py::train_hier --stage1-checkpoint <path>"
        )

    override_list = [o.strip() for o in overrides.split(",") if o.strip()]
    if dry_run:
        override_list += [
            "stage2.n_epochs=2",
            "loader.batch_size=8",
            "wandb.enabled=False",
        ]

    print(f"[local] submitting hierarchical job — image: {_tag}, checkpoint: {stage1_checkpoint}")
    print(f"[local] overrides: {override_list}")
    train_hierarchical.remote(
        stage1_checkpoint=stage1_checkpoint,
        data=data,
        overrides=override_list,
    )
