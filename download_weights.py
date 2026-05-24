#!/usr/bin/env python3
"""Download LeWM paper weights from HuggingFace into baseline_paper/.

Model repos (https://huggingface.co/quentinll):
    pusht     quentinll/lewm-pusht
    tworooms  quentinll/lewm-tworooms
    cube      quentinll/lewm-cube
    reacher   quentinll/lewm-reacher

Each env downloads weights.pt and config.json into:
    baseline_paper/<env>/weights.pt
    baseline_paper/<env>/config.json

The script is idempotent — already-downloaded files are skipped.

Usage:
    python download_weights.py                      # all four envs
    python download_weights.py --only tworooms      # single env
    python download_weights.py --only cube pusht    # multiple envs
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency bootstrap
# ---------------------------------------------------------------------------

def _ensure_deps() -> None:
    try:
        __import__("huggingface_hub")
    except ImportError:
        print("[bootstrap] installing: huggingface_hub", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--break-system-packages", "-q", "huggingface_hub"]
        )


_ensure_deps()

from huggingface_hub import hf_hub_download  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent
DEST_ROOT = REPO_ROOT / "baseline_paper"

MODEL_REPOS: dict[str, str] = {
    "pusht":    "quentinll/lewm-pusht",
    "tworooms": "quentinll/lewm-tworooms",
    "cube":     "quentinll/lewm-cube",
    "reacher":  "quentinll/lewm-reacher",
}

FILES = ["weights.pt", "config.json"]

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_weights(env: str, repo_id: str) -> list[Path]:
    dest_dir = DEST_ROOT / env
    dest_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    for filename in FILES:
        dest = dest_dir / filename
        if dest.exists():
            print(f"  [skip] {env}/{filename} already present", flush=True)
            downloaded.append(dest)
            continue

        print(f"  [download] {repo_id}/{filename} -> {dest}", flush=True)
        t0 = time.time()
        tmp = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
        )
        import shutil
        shutil.copy2(tmp, dest)
        elapsed = time.time() - t0
        size_mb = dest.stat().st_size / 1e6
        print(f"  [done] {env}/{filename} ({size_mb:.1f} MB, {elapsed:.1f}s)", flush=True)
        downloaded.append(dest)

    return downloaded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(MODEL_REPOS),
        metavar="ENV",
        help="One or more envs to download (pusht, tworooms, cube, reacher). Defaults to all.",
    )
    args = parser.parse_args()

    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Destination: {DEST_ROOT}\n")

    envs = args.only or list(MODEL_REPOS)
    results: dict[str, list[Path]] = {}
    for env in envs:
        print(f"\n[{env}] {MODEL_REPOS[env]}")
        try:
            results[env] = download_weights(env=env, repo_id=MODEL_REPOS[env])
        except Exception as exc:
            print(f"  [error] {exc!r}", flush=True)
            results[env] = []

    print("\n=== Summary ===")
    for env, paths in results.items():
        for p in paths:
            size_mb = p.stat().st_size / 1e6 if p.exists() else 0
            print(f"  {env:10s}  {p.relative_to(REPO_ROOT)}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
