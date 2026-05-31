"""Precompute frozen JEPA frame embeddings for stage-2 hierarchical training.

Iterates every raw frame in the HDF5 dataset, encodes it with the frozen stage-1
JEPA, and saves a numpy array of shape (N_raw_frames, embed_dim) to the same
directory as the HDF5 file.  train_hierarchical.py loads this file when
``stage2.use_cached_embeddings: true``, eliminating the ViT forward pass from
every training step.

The output is keyed only to the dataset name and img_size, so it can be reused
across multiple stage-2 runs that share the same data and stage-1 encoder.

Usage
-----
python precompute_embeddings.py stage1_checkpoint=<path>
python precompute_embeddings.py stage1_checkpoint=<path> data=pusht
python precompute_embeddings.py stage1_checkpoint=<path> precompute_batch_size=512 device=cpu
"""

import logging
import time
from pathlib import Path

import h5py
import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, open_dict
from torchvision.transforms import v2 as T

py_log = logging.getLogger(__name__)


def _make_transform(img_size: int) -> T.Compose:
    stats = spt.data.dataset_stats.ImageNet
    return T.Compose([
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=stats["mean"], std=stats["std"]),
        T.Resize(size=img_size, antialias=True),
    ])


@hydra.main(version_base=None, config_path="./config/train", config_name="hierarchical")
def run(cfg: DictConfig) -> None:
    t_script = time.perf_counter()
    # allow extra keys not present in the base config
    with open_dict(cfg):
        device = cfg.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
        batch_size = cfg.setdefault("precompute_batch_size", 256)

    cache_dir = Path(cfg.get("cache_dir") or swm.data.utils.get_cache_dir())
    h5_path = cache_dir / f"{cfg.data.dataset.name}.h5"
    out_path = cache_dir / f"{cfg.data.dataset.name}_emb_img{cfg.img_size}.npy"

    py_log.info(
        "Precomputing embeddings — dataset=%s  img_size=%d  device=%s  batch=%d",
        cfg.data.dataset.name, cfg.img_size, device, batch_size,
    )
    py_log.info("HDF5 source : %s", h5_path)
    py_log.info("Output      : %s", out_path)

    if out_path.exists():
        py_log.info("Output already exists — delete it to force recompute.")
        return

    py_log.info("Loading stage-1 checkpoint from %s", cfg.stage1_checkpoint)
    jepa = torch.load(cfg.stage1_checkpoint, map_location=device, weights_only=False)
    jepa = jepa.eval().to(device)

    transform = _make_transform(cfg.img_size)

    with h5py.File(h5_path, "r", swmr=True) as f:
        n_total = int(f["pixels"].shape[0])
        py_log.info("Raw frames in dataset: %d", n_total)

        # Derive embed_dim without hardcoding it.
        dummy = torch.from_numpy(f["pixels"][:1]).permute(0, 3, 1, 2)  # (1, C, H, W)
        dummy = transform(dummy).unsqueeze(1).to(device)                # (1, 1, C, H, W)
        with torch.no_grad():
            embed_dim = jepa.encode({"pixels": dummy})["emb"].shape[-1]
        py_log.info("embed_dim: %d", embed_dim)

        all_emb = np.zeros((n_total, embed_dim), dtype=np.float32)

        t0 = time.perf_counter()
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            raw = f["pixels"][start:end]                              # (B, H, W, C) uint8
            frames = torch.from_numpy(raw).permute(0, 3, 1, 2)       # (B, C, H, W) uint8
            frames = transform(frames).unsqueeze(1).to(device)        # (B, 1, C, H, W)
            with torch.no_grad():
                emb = jepa.encode({"pixels": frames})["emb"]          # (B, 1, D)
            all_emb[start:end] = emb[:, 0].cpu().numpy()

            if (start // batch_size) % 20 == 0 or end == n_total:
                pct = 100.0 * end / n_total
                py_log.info("  %.1f%%  (%d/%d frames)  %.1fs", pct, end, n_total, time.perf_counter() - t0)

    np.save(out_path, all_emb)
    encoding_s = time.perf_counter() - t0
    total_s = time.perf_counter() - t_script
    py_log.info(
        "Saved %s — encoding: %.1f s  total (incl. model load): %.1f s (%.1f min)",
        out_path, encoding_s, total_s, total_s / 60,
    )


if __name__ == "__main__":
    run()
