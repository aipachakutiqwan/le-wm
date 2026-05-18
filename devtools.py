#!/usr/bin/env python3
"""Devtools for LeWorldModel — build, push, and manage training images."""

import logging
import os
import re
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
IMAGE_NAME = "cs231n_project/lewm"  # fixed local name — only tag should change between runs
IMAGE_TAG = "latest"


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
            log.info("Base image not found locally, pulling: %s", BASE_IMAGE)
            self._run(["docker", "pull", BASE_IMAGE])
        else:
            log.info("Base image already present: %s", BASE_IMAGE)

        start = time.monotonic()
        self._run([
            "docker", "build",
            "-f", str(REPO_ROOT / "cloud" / "Dockerfile"),
            "-t", f"{IMAGE_NAME}:{tag}",
            str(REPO_ROOT),
        ])
        elapsed = time.monotonic() - start
        log.info("Built: %s:%s in %dm %ds", IMAGE_NAME, tag, elapsed // 60, elapsed % 60)

        if push:
            self.login(username, registry)
            self.push_docker(tag, username, registry)

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

    def push_docker(self, tag: str, username: str = None, registry: str = "ghcr.io") -> None:
        """Tag the local image and push it to the registry."""
        username = self._github_username(username)
        image_name = IMAGE_NAME.split("/")[-1]  # strip local namespace, use just "lewm"
        remote = f"{registry}/{username}/{image_name}:{tag}"
        self._run(["docker", "tag", f"{IMAGE_NAME}:{tag}", remote])
        self._run(["docker", "push", remote])
        log.info("Pushed: %s", remote)

    def pull_docker(self, tag: str, username: str = None, registry: str = "ghcr.io") -> None:
        """Pull the training image from the registry and retag it locally."""
        username = self._github_username(username)
        image_name = IMAGE_NAME.split("/")[-1]
        remote = f"{registry}/{username}/{image_name}:{tag}"
        self._run(["docker", "pull", remote])
        self._run(["docker", "tag", remote, f"{IMAGE_NAME}:{tag}"])
        log.info("Pulled and tagged as %s:%s", IMAGE_NAME, tag)

    def _base_run_flags(self) -> list[str]:
        stablewm_home = os.environ.get("STABLEWM_HOME")
        if not stablewm_home:
            raise EnvironmentError("STABLEWM_HOME environment variable is not set")

        flags = [
            "--rm", "--gpus", "all",
            "--ipc=host",  # share host /dev/shm — avoids OOM from DataLoader pinned memory workers
            "-v", f"{stablewm_home}:/stablewm-home",
            "-e", "STABLEWM_HOME=/stablewm-home",
        ]
        if wandb_key := os.environ.get("WANDB_API_KEY"):
            flags += ["-e", f"WANDB_API_KEY={wandb_key}"]
        return flags

    def run_local(self, tag: str, data: str = "tworoom", setup: str = None) -> None:
        """Run training locally inside the Docker container."""
        cmd = ["docker", "run"] + self._base_run_flags() + [f"{IMAGE_NAME}:{tag}", f"data={data}"]
        if setup:
            cmd += [f"setup={setup}"]
        self._run(cmd)

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
