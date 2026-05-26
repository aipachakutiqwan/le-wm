#!/usr/bin/env python3
"""Devtools for LeWorldModel — build, push, and manage training images."""

import logging
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import fire
except ImportError:
    import shutil
    uv = shutil.which("uv")
    if uv:
        subprocess.run([uv, "pip", "install", "fire"], check=True)
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", "fire", "--user"], check=True)
    os.execv(sys.executable, [sys.executable] + sys.argv)  # restart with fire now installed

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent
BASE_IMAGE = "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04"
# GPU training targets x86_64 Linux. Building on Apple Silicon without this defaults to
# linux/arm64, where box2d (stable-worldmodel[env]) has no wheels.
DOCKER_PLATFORM = "linux/amd64"
IMAGE_NAME = "cs231n_project/lewm"  # fixed local name — only tag should change between runs
IMAGE_TAG = "latest"
GHCR_IMAGE_OWNER = "jadhavan"  # default GHCR namespace — override with GHCR_IMAGE_OWNER env var


class DevTools:

    def _git_tag(self) -> str:
        def git(*args) -> str:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, check=True, cwd=REPO_ROOT,
            ).stdout.strip()

        date = datetime.now().strftime("%Y%m%d")
        git_hash = git("rev-parse", "--short", "HEAD")
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        branch = re.sub(r"[^a-zA-Z0-9]", "-", branch)  # replace / and special chars
        return f"{date}_{git_hash}_{branch}"

    def _run(self, cmd: list[str]) -> None:
        log.info(" ".join(cmd))
        subprocess.run(cmd, check=True)

    def _image_exists_locally(self, image: str) -> bool:
        result = subprocess.run(
            ["docker", "images", "-q", image],
            capture_output=True, text=True,
        )
        return bool(result.stdout.strip())

    def build_docker(self, tag: str = None, push: bool = False, username: str = None, registry: str = "ghcr.io") -> None:
        """Pull base image if needed, then build the training image."""
        tag = tag or self._git_tag()

        if not self._image_exists_locally(BASE_IMAGE):
            log.info("Base image not found locally, pulling: %s (%s)", BASE_IMAGE, DOCKER_PLATFORM)
            self._run(["docker", "pull", "--platform", DOCKER_PLATFORM, BASE_IMAGE])
        else:
            log.info("Base image already present: %s", BASE_IMAGE)

        start = time.monotonic()
        self._run([
            "docker", "build",
            "--platform", DOCKER_PLATFORM,
            "-f", str(REPO_ROOT / "cloud" / "Dockerfile"),
            "-t", f"{IMAGE_NAME}:{tag}",
            str(REPO_ROOT),
        ])
        elapsed = time.monotonic() - start
        log.info("Built: %s:%s in %dm %ds", IMAGE_NAME, tag, elapsed // 60, elapsed % 60)

        if push:
            self.login(username=username, registry=registry)
            self.push_docker(tag=tag, username=username, registry=registry)

    def _github_username(self, username: str = None) -> str:
        resolved = username or os.environ.get("GITHUB_USERNAME")
        if not resolved:
            raise EnvironmentError("GITHUB_USERNAME environment variable is not set")
        return resolved

    def login(self, username: str = None, registry: str = "ghcr.io") -> None:
        """Log in to a container registry using GITHUB_PAT and GITHUB_USERNAME from the environment."""
        username = self._github_username(username)
        pat = os.environ.get("GITHUB_PAT")
        if not pat:
            raise EnvironmentError("GITHUB_PAT environment variable is not set")
        subprocess.run(
            ["docker", "login", registry, "-u", username, "--password-stdin"],
            input=pat, text=True, check=True,
        )
        log.info("Logged in to %s as %s", registry, username)

    def push_docker(self, tag: str, username: str = None, owner: str = None, registry: str = "ghcr.io") -> None:
        """Tag the local image and push it to the registry in OCI format.

        Uses skopeo (if available) to push OCI-format layers, which are required
        by Modal's umoci image builder. Falls back to docker push (Docker format)
        with a warning when skopeo is not installed.

        username: your GitHub username for auth (GITHUB_USERNAME env var)
        owner:    GitHub username to push under (GHCR_IMAGE_OWNER env var) — defaults to username
        """
        username = self._github_username(username)
        image_owner = owner or os.environ.get("GHCR_IMAGE_OWNER") or GHCR_IMAGE_OWNER
        image_name = IMAGE_NAME.split("/")[-1]
        remote = f"{registry}/{image_owner}/{image_name}:{tag}"
        local = f"{IMAGE_NAME}:{tag}"

        if shutil.which("skopeo"):
            # skopeo reads credentials from ~/.docker/config.json (populated by docker login)
            self._run([
                "skopeo", "copy", "--format", "oci",
                f"docker-daemon:{local}",
                f"docker://{remote}",
            ])
        else:
            log.warning(
                "skopeo not found — pushing with docker push (Docker format). "
                "Modal may reject the image. Install skopeo for OCI-format push."
            )
            self._run(["docker", "tag", local, remote])
            self._run(["docker", "push", remote])

        log.info("Pushed: %s", remote)

    def pull_docker(self, tag: str, username: str = None, owner: str = None, registry: str = "ghcr.io") -> None:
        """Pull the training image from the registry and retag it locally.
        username: your GitHub username for auth (GITHUB_USERNAME env var)
        owner:    GitHub username who pushed the image (GHCR_IMAGE_OWNER env var) — defaults to username
        """
        username = self._github_username(username)
        image_owner = owner or os.environ.get("GHCR_IMAGE_OWNER") or GHCR_IMAGE_OWNER
        image_name = IMAGE_NAME.split("/")[-1]
        remote = f"{registry}/{image_owner}/{image_name}:{tag}"
        self._run(["docker", "pull", remote])
        self._run(["docker", "tag", remote, f"{IMAGE_NAME}:{tag}"])
        log.info("Pulled and tagged as %s:%s", IMAGE_NAME, tag)

    def _base_run_flags(self) -> list[str]:
        stablewm_home = os.environ.get("STABLEWM_HOME")
        if not stablewm_home:
            raise EnvironmentError("STABLEWM_HOME environment variable is not set")

        flags = [
            "--rm", "--gpus", "all", "--platform", DOCKER_PLATFORM,
            "--ipc=host",  # share host /dev/shm — avoids OOM from DataLoader pinned memory workers
            "-v", f"{stablewm_home}:/stablewm-home",
            "-e", "STABLEWM_HOME=/stablewm-home",
        ]
        if wandb_key := os.environ.get("WANDB_API_KEY"):
            flags += ["-e", f"WANDB_API_KEY={wandb_key}"]
        # Reduces CUDA memory fragmentation: reserved-but-unallocated blocks can
        # be reused for non-contiguous allocations instead of triggering OOM.
        # Hit this on L4 (22 GiB usable) during ViT MLP forward at batch_size=256.
        # https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
        # flags += ["-e", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]
        return flags

    def run_local(self, tag: str, data: str = "tworoom", setup: str = None, overrides: list = None) -> None:
        """Run training locally inside the Docker container."""
        cmd = ["docker", "run"] + self._base_run_flags() + [f"{IMAGE_NAME}:{tag}", f"data={data}"]
        if setup:
            cmd += [f"setup={setup}"]
        if overrides:
            if isinstance(overrides, str):
                overrides = overrides.strip("[]").split(",")
            cmd += overrides
        self._run(cmd)

    def download_modal(self, *envs: str) -> None:
        """Download datasets from HuggingFace directly into the Modal volume.

        Much faster than uploading from local — Modal servers pull at datacenter speed.
        Run once per dataset; reused across all training runs.

        Examples:
            ./devtools.py download_modal tworoom
            ./devtools.py download_modal tworoom pusht
        """
        if not envs:
            envs = ("tworoom",)
        self._run([
            "modal", "run", f"{REPO_ROOT / 'cloud' / 'modal_train.py'}::download",
            "--envs", ",".join(envs),
        ])

    def run_modal(self, tag: str, data: str = "tworoom", setup: str = "cloud_a10g",
                  overrides: list = None, dry_run: bool = False) -> None:
        """Submit a training job to Modal using the GHCR image.

        Requires: pip install modal && modal setup
        One-time volume/secret setup: see cloud/modal_train.py docstring.
        """
        overrides_str = ""
        if overrides:
            if isinstance(overrides, str):
                overrides = overrides.strip("[]").split(",")
            overrides_str = ",".join(overrides)

        cmd = [
            "modal", "run", str(REPO_ROOT / "cloud" / "modal_train.py"),
            "--data", data,
            "--setup", setup,
        ]
        if overrides_str:
            cmd += ["--overrides", overrides_str]
        if dry_run:
            cmd += ["--dry-run"]

        # LEWM_TAG is read at module import time by modal_train.py
        env = {**os.environ, "LEWM_TAG": tag}
        log.info("LEWM_TAG=%s %s", tag, " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)

    def run_hierarchical_local(self, tag: str, stage1_checkpoint: str,
                               data: str = "tworoom", setup: str = None,
                               overrides: list = None, dry_run: bool = False) -> None:
        """Run stage-2 hierarchical training locally inside the Docker container.

        stage1_checkpoint: path inside the container. If the checkpoint lives in
                           STABLEWM_HOME it will be at /stablewm-home/<filename>.ckpt.
                           Alternatively use /app/baseline/... for Git-tracked files.
        """
        override_list = []
        if overrides:
            if isinstance(overrides, str):
                override_list = overrides.strip("[]").split(",")
            else:
                override_list = list(overrides)

        if dry_run:
            override_list += [
                "stage2.n_epochs=2",
                "loader.batch_size=8",
                "wandb.enabled=False",
            ]

        cmd = [
            "docker", "run",
            "--entrypoint", "python",
        ] + self._base_run_flags() + [
            f"{IMAGE_NAME}:{tag}",
            "train_hierarchical.py",
            f"stage1_checkpoint={stage1_checkpoint}",
            f"data={data}",
        ]
        if setup:
            cmd += [f"setup={setup}"]
        cmd += override_list
        self._run(cmd)

    def run_hierarchical_modal(self, tag: str, stage1_checkpoint: str,
                               data: str = "tworoom", overrides: list = None,
                               dry_run: bool = False) -> None:
        """Submit a stage-2 hierarchical training job to Modal (A100 GPU).

        The stage-1 checkpoint must already be in the Modal volume.
        Upload it first if needed:
            modal volume put lewm-data <local>_object.ckpt <filename>_object.ckpt

        stage1_checkpoint: absolute path inside /stablewm-home in the Modal volume.
                           e.g. /stablewm-home/lewm_epoch_100_object.ckpt

        Requires: pip install modal && modal setup
        """
        overrides_str = ""
        if overrides:
            if isinstance(overrides, str):
                overrides = overrides.strip("[]").split(",")
            overrides_str = ",".join(overrides)

        cmd = [
            "modal", "run",
            f"{REPO_ROOT / 'cloud' / 'modal_train.py'}::train_hier",
            "--stage1-checkpoint", stage1_checkpoint,
            "--data", data,
        ]
        if overrides_str:
            cmd += ["--overrides", overrides_str]
        if dry_run:
            cmd += ["--dry-run"]

        env = {**os.environ, "LEWM_TAG": tag}
        log.info("LEWM_TAG=%s %s", tag, " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)

    def eval_modal(self, tag: str, policy: str, config: str = "tworoom", overrides: list = None) -> None:
        """Submit an eval job to Modal using the GHCR image (A10G GPU).

        policy: path to checkpoint directory or stem inside /stablewm-home
                e.g. /stablewm-home/lewm_epoch_9  or  /stablewm-home/<run_id>

        Requires: pip install modal && modal setup
        """
        overrides_str = ""
        if overrides:
            if isinstance(overrides, str):
                overrides = overrides.strip("[]").split(",")
            overrides_str = ",".join(overrides)

        cmd = [
            "modal", "run", f"{REPO_ROOT / 'cloud' / 'modal_train.py'}::eval",
            "--policy", policy,
            "--config", config,
        ]
        if overrides_str:
            cmd += ["--overrides", overrides_str]

        env = {**os.environ, "LEWM_TAG": tag}
        log.info("LEWM_TAG=%s %s", tag, " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)

    def dev(self, tag: str) -> None:
        """Launch a bash shell with the local repo mounted at /app.
        Edits to the repo on the host are instantly reflected inside the container.
        """
        cmd = [
            "docker", "run", "-it",
            "-v", f"{REPO_ROOT}:/app",   # live repo mount — overrides baked-in code
        ] + self._base_run_flags() + ["--entrypoint", "bash", f"{IMAGE_NAME}:{tag}"]
        self._run(cmd)


if __name__ == "__main__":
    fire.Fire(DevTools)
