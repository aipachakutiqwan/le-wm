"""Pre-compute encoder embeddings for fast stage-2 training.

Loads the frozen JEPA encoder from a stage-1 checkpoint, encodes all frames
in the source HDF5 dataset, and writes a new HDF5 file with embeddings replacing
the pixel data.  The result is a drop-in replacement for the original dataset
that skips the encoder forward pass during training.

Usage
-----
python precompute_embeddings.py stage1_checkpoint=<path>
python precompute_embeddings.py stage1_checkpoint=<path> data=ogb
python precompute_embeddings.py stage1_checkpoint=<path> precompute.device=cuda precompute.batch_size=512
"""

import logging
from pathlib import Path

py_log = logging.getLogger(__name__)

import h5py
import hdf5plugin  # noqa: F401
import hydra
import numpy as np
import torch
from torchvision.transforms import v2 as TV

import stable_worldmodel as swm


@hydra.main(version_base=None, config_path="./config/train", config_name="hierarchical")
def run(cfg):
    device_str = cfg.precompute.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    batch_size = cfg.precompute.batch_size
    img_size = cfg.img_size

    datasets_dir = swm.data.utils.get_cache_dir(sub_folder="datasets")
    name = cfg.data.dataset.name
    src_path = datasets_dir / f"{name}.h5"
    dst_path = datasets_dir / f"{name}_emb.h5"
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    py_log.info("Source:      %s", src_path)
    py_log.info("Destination: %s", dst_path)
    py_log.info("Device:      %s  batch_size: %d", device, batch_size)

    py_log.info("Loading stage-1 checkpoint from %s", cfg.stage1_checkpoint)
    jepa = torch.load(cfg.stage1_checkpoint, map_location="cpu", weights_only=False)
    jepa.eval().to(device)

    # Replicate the training transform pipeline (ToImage + Resize from utils.get_img_preprocessor)
    transform = TV.Compose([
        TV.ToDtype(torch.float32, scale=True),
        TV.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        TV.Resize(size=img_size, interpolation=2, antialias=True),
    ])

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        ep_len = src["ep_len"][:]
        ep_offset = src["ep_offset"][:]
        dst.create_dataset("ep_len", data=ep_len)
        dst.create_dataset("ep_offset", data=ep_offset)

        for key in src.keys():
            if key in ("ep_len", "ep_offset", "pixels"):
                continue
            py_log.info("Copying column '%s' ...", key)
            dst.create_dataset(key, data=src[key][:], **hdf5plugin.LZ4())

        total_frames = src["pixels"].shape[0]
        embed_dim = cfg.wm.embed_dim
        py_log.info("Encoding %d frames → emb dim=%d ...", total_frames, embed_dim)

        emb_ds = dst.create_dataset(
            "emb",
            shape=(total_frames, embed_dim),
            dtype="float32",
            **hdf5plugin.LZ4(),
        )

        for start in range(0, total_frames, batch_size):
            end = min(start + batch_size, total_frames)
            # pixels stored as (N, H, W, C) uint8
            frames = src["pixels"][start:end]
            x = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2)  # (B, C, H, W)
            x = transform(x).to(device)
            with torch.no_grad():
                emb = jepa.encode({"pixels": x.unsqueeze(1)})["emb"]  # (B, 1, D)
                emb = emb[:, 0].cpu().float().numpy()                  # (B, D)
            emb_ds[start:end] = emb
            if (start // batch_size) % 10 == 0:
                py_log.info("  %d / %d frames encoded", end, total_frames)

    py_log.info("Done — embedding dataset written to %s", dst_path)


if __name__ == "__main__":
    run()
