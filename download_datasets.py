#!/usr/bin/env python3
"""Download and extract LeWM datasets from HuggingFace into STABLEWM_HOME.

Datasets (https://huggingface.co/collections/quentinll/lewm):
    pusht      quentinll/lewm-pusht
    cube       quentinll/lewm-cube
    tworooms   quentinll/lewm-tworooms
    reacher    quentinll/lewm-reacher

Archives are downloaded via huggingface_hub, then extracted to $STABLEWM_HOME
(defaults to ~/.stable-wm). The script is idempotent — already-extracted files
are skipped.

Usage:
    python download_datasets.py                      # all four envs
    python download_datasets.py --only tworooms      # single env
    python download_datasets.py --only cube pusht    # multiple envs
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency bootstrap
# ---------------------------------------------------------------------------

def _ensure_deps() -> None:
    missing = []
    for pkg, mod in [("huggingface_hub", "huggingface_hub"), ("zstandard", "zstandard")]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[bootstrap] installing: {missing}", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--break-system-packages", "-q", *missing]
        )


_ensure_deps()

import zstandard as zstd  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_REPOS: dict[str, str] = {
    "pusht":    "quentinll/lewm-pusht",
    "cube":     "quentinll/lewm-cube",
    "tworooms": "quentinll/lewm-tworooms",
    "reacher":  "quentinll/lewm-reacher",
}

STABLEWM_HOME = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable-wm")).expanduser()
DOWNLOAD_ROOT = STABLEWM_HOME / "hf_cache"

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_repo(env: str, repo_id: str) -> Path:
    local_dir = DOWNLOAD_ROOT / env
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[download] {repo_id} -> {local_dir}", flush=True)
    t0 = time.time()
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
    )
    print(f"[download] done in {time.time() - t0:.1f}s", flush=True)
    return Path(path)

# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------

def _extract_safely(tar: tarfile.TarFile, dest_dir: Path) -> None:
    dest_abs = dest_dir.resolve()
    for member in tar:
        target = (dest_dir / member.name).resolve()
        if not str(target).startswith(str(dest_abs)):
            raise RuntimeError(f"unsafe tar member: {member.name}")
        if target.exists() and member.isfile():
            print(f"    [skip] {member.name} already present", flush=True)
            continue
        tar.extract(member=member, path=dest_dir)
        if member.isfile():
            print(f"    [write] {member.name} ({target.stat().st_size / 1e9:.2f} GB)", flush=True)


def extract_archive(archive_path: Path, dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = archive_path.name.lower()
    print(f"  [extract] {archive_path.name}", flush=True)

    if name.endswith(".tar.zst") or name.endswith(".tzst"):
        dctx = zstd.ZstdDecompressor()
        with open(archive_path, "rb") as fh, dctx.stream_reader(fh) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                _extract_safely(tar, dest_dir)
        return sorted(dest_dir.glob("*.h5"))

    if name.endswith(".zst"):
        inner_name = archive_path.name[: -len(".zst")]
        target = dest_dir / inner_name
        if target.exists():
            print(f"    [skip] {target.name} already present", flush=True)
        else:
            dctx = zstd.ZstdDecompressor()
            with open(archive_path, "rb") as src, open(target, "wb") as dst:
                dctx.copy_stream(src, dst)
            print(f"    [write] {target} ({target.stat().st_size / 1e9:.2f} GB)", flush=True)
        return sorted(dest_dir.glob("*.h5"))

    if name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2")):
        with tarfile.open(archive_path, "r:*") as tar:
            _extract_safely(tar, dest_dir)
        return sorted(dest_dir.glob("*.h5"))

    if name.endswith(".h5"):
        target = dest_dir / archive_path.name
        if not target.exists():
            shutil.copy2(archive_path, target)
        return sorted(dest_dir.glob("*.h5"))

    return []


def _predicted_h5(archive_path: Path, dest_dir: Path) -> list[Path] | None:
    name = archive_path.name.lower()
    if name.endswith(".zst") and not (name.endswith(".tar.zst") or name.endswith(".tzst")):
        return [dest_dir / archive_path.name[: -len(".zst")]]
    if name.endswith(".h5"):
        return [dest_dir / archive_path.name]
    return None  # tar variants — can't predict member names without opening


# ---------------------------------------------------------------------------
# Stage one env
# ---------------------------------------------------------------------------

def stage_dataset(env: str, repo_id: str, download: bool = True, extract: bool = True) -> list[Path]:
    if download:
        snapshot = download_repo(env=env, repo_id=repo_id)
    else:
        snapshot = DOWNLOAD_ROOT / env
        if not snapshot.exists():
            raise FileNotFoundError(f"No cached download found at {snapshot}. Run without --extract-only first.")

    if not extract:
        print(f"  [skip-extract] {env}: download only", flush=True)
        return []

    archives: list[Path] = []
    for ext in ("*.tar.zst", "*.tzst", "*.h5.zst", "*.zst", "*.tar.gz", "*.tgz", "*.tar"):
        archives.extend(snapshot.rglob(ext))
    archives = sorted({a.resolve() for a in archives})

    h5_files: list[Path] = []
    if archives:
        for archive in archives:
            predicted = _predicted_h5(archive_path=archive, dest_dir=STABLEWM_HOME)
            if predicted is not None and predicted and all(p.exists() for p in predicted):
                print(f"  [skip-extract] {archive.name} (outputs already present)", flush=True)
                h5_files.extend(predicted)
            else:
                h5_files.extend(extract_archive(archive_path=archive, dest_dir=STABLEWM_HOME))
    else:
        for h5 in snapshot.rglob("*.h5"):
            target = STABLEWM_HOME / h5.name
            if target.exists():
                print(f"  [skip-copy] {target.name} already present", flush=True)
            else:
                shutil.copy2(h5, target)
            h5_files.append(target)

    h5_files = sorted({p.resolve() for p in h5_files})
    print(f"  [done] {env}: {len(h5_files)} .h5 file(s) ready in {STABLEWM_HOME}", flush=True)
    return h5_files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(DATASET_REPOS),
        metavar="ENV",
        help="One or more envs to process (pusht, cube, tworooms, reacher). Defaults to all.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--download-only",
        action="store_true",
        help="Download archives to hf_cache but do not extract.",
    )
    mode.add_argument(
        "--extract-only",
        action="store_true",
        help="Extract from already-downloaded hf_cache; skip downloading.",
    )
    args = parser.parse_args()

    STABLEWM_HOME.mkdir(parents=True, exist_ok=True)
    print(f"STABLEWM_HOME = {STABLEWM_HOME}")

    do_download = not args.extract_only
    do_extract = not args.download_only

    envs = args.only or list(DATASET_REPOS)
    results: dict[str, list[Path]] = {}
    for env in envs:
        try:
            results[env] = stage_dataset(env=env, repo_id=DATASET_REPOS[env], download=do_download, extract=do_extract)
        except Exception as exc:
            print(f"[error] {env}: {exc!r}", flush=True)
            results[env] = []

    print("\n=== Summary ===")
    for env, paths in results.items():
        for p in paths:
            size_gb = p.stat().st_size / 1e9 if p.exists() else 0
            print(f"  {env:10s}  {p.name}  ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
