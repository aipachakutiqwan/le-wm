"""Stage-2 training script for HierarchicalLeWM.

Stage 1 (JEPA) is trained by train.py.  This script loads the resulting
checkpoint, wraps it in HierarchicalLeWM, and runs the stage-2 teacher-
forcing loop that jointly trains A_ψ and P^(2).

On first run the frozen JEPA encoder is applied once to every raw frame in the
HDF5 dataset and the result is saved to disk.  Subsequent runs reuse the cached
file, so the ViT forward pass is never repeated during training.

Usage
-----
# Single GPU
python train_hierarchical.py stage1_checkpoint=<path/to/weights.pt>

# Two A100s (DDP) — torchrun handles rank/world-size env vars
torchrun --nproc_per_node=2 train_hierarchical.py stage1_checkpoint=<path>

# Different dataset
python train_hierarchical.py data=pusht stage1_checkpoint=<path>

# Quick smoke-test
python train_hierarchical.py stage1_checkpoint=<path> \\
    stage2.n_epochs=2 loader.batch_size=8 wandb.enabled=False
"""

import os
import logging
import time
from pathlib import Path

py_log = logging.getLogger(__name__)

import h5py
import hydra
import numpy as np
import wandb
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf, open_dict
from torchvision.transforms import v2 as T

# DataLoader workers pass tensors via /dev/shm by default. Modal containers cap
# /dev/shm small, so multi-worker loading with large batches blows it out with
# "No space left on device". file_system strategy uses regular tmpfs/RAM instead.
torch.multiprocessing.set_sharing_strategy("file_system")

from hierarchical_lewm import HierarchicalLeWM, train_hierarchical_lewm
from utils import get_column_normalizer, ModelObjectCallBack


# ──────────────────────────────────────────────────────────────────────────────
# Embedding cache
# ──────────────────────────────────────────────────────────────────────────────


def _ensure_embeddings(
    jepa: torch.nn.Module,
    h5_path: Path,
    out_path: Path,
    img_size: int,
    device: str,
    batch_size: int = 1024,
) -> np.ndarray:
    """Return cached embeddings, computing and saving them first if needed.

    The cache file is keyed to both the dataset and the stage-1 checkpoint stem
    so switching checkpoints never produces stale embeddings.
    """
    if out_path.exists():
        py_log.info("Reusing cached embeddings from %s", out_path)
        return np.load(out_path, mmap_mode="r")

    py_log.info("Embedding cache not found — computing from %s …", h5_path)
    jepa.to(device).eval()
    stats = spt.data.dataset_stats.ImageNet
    transform = T.Compose([
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=stats["mean"], std=stats["std"]),
        T.Resize(size=img_size, antialias=True),
    ])

    t0 = time.perf_counter()
    with h5py.File(h5_path, "r", swmr=True) as f:
        n_total = int(f["pixels"].shape[0])
        py_log.info("Raw frames to encode: %d", n_total)
        all_emb = None

        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            raw = f["pixels"][start:end]                        # (B, H, W, C) uint8
            frames = torch.from_numpy(raw).permute(0, 3, 1, 2) # (B, C, H, W)
            frames = transform(frames).unsqueeze(1).to(device)  # (B, 1, C, H, W)
            with torch.no_grad():
                emb = jepa.encode({"pixels": frames})["emb"][:, 0]  # (B, D)
            if all_emb is None:
                all_emb = np.zeros((n_total, emb.shape[-1]), dtype=np.float32)
            all_emb[start:end] = emb.float().cpu().numpy()

            if (start // batch_size) % 20 == 0 or end == n_total:
                py_log.info(
                    "  %.1f%%  (%d/%d)  %.1fs",
                    100.0 * end / n_total, end, n_total, time.perf_counter() - t0,
                )

    tmp = out_path.with_suffix(f".{os.getpid()}.tmp.npy")
    np.save(tmp, all_emb)
    tmp.rename(out_path)
    py_log.info("Embeddings saved to %s  (%.1fs)", out_path, time.perf_counter() - t0)
    return np.load(out_path, mmap_mode="r")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


@hydra.main(version_base=None, config_path="./config/train", config_name="hierarchical")
def run(cfg):
    t_run = time.perf_counter()

    # ── Distributed setup ─────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank >= 0
    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = cfg.device
    is_main = not is_distributed or dist.get_rank() == 0
    # ──────────────────────────────────────────────────────────────────────────

    py_log.info(
        "Hierarchical stage-2 training — data=%s checkpoint=%s  rank=%s/%s",
        cfg.data.dataset.name,
        cfg.stage1_checkpoint,
        dist.get_rank() if is_distributed else 0,
        dist.get_world_size() if is_distributed else 1,
    )

    #########################
    ##       dataset       ##
    #########################

    with open_dict(cfg):
        cfg.data.dataset.num_steps = cfg.stage2_num_steps

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)

    ##############################
    ##       model / JEPA       ##
    ##############################

    py_log.info("STABLEWM_HOME set to %s", os.getenv("STABLEWM_HOME"))
    py_log.info("Loading stage-1 checkpoint from %s", cfg.stage1_checkpoint)
    jepa = torch.load(cfg.stage1_checkpoint, map_location=device, weights_only=False)
    jepa.eval()

    ##############################
    ##   embedding cache        ##
    ##############################

    cache_dir = Path(cfg.get("cache_dir") or swm.data.utils.get_cache_dir())
    ckpt_stem = Path(cfg.stage1_checkpoint).stem
    emb_path = cache_dir / f"{cfg.data.dataset.name}_{ckpt_stem}_img{cfg.img_size}_emb.npy"
    # Only rank 0 computes the cache; others wait at the barrier then load from disk.
    if is_main:
        _ensure_embeddings(
            jepa, cache_dir / f"{cfg.data.dataset.name}.h5", emb_path,
            cfg.img_size, device,
            batch_size=cfg.get("cache_batch_size", 1024),
        )
    if is_distributed:
        dist.barrier()
    emb_array = np.load(emb_path, mmap_mode="r")
    dataset._cache["emb"] = emb_array
    dataset._keys = ["emb" if k == "pixels" else k for k in dataset._keys]

    ##############################
    ##     transforms           ##
    ##############################

    # Image preprocessing is skipped — embeddings are already encoded.
    transforms = []
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    dataset.transform = spt.data.transforms.Compose(*transforms) if transforms else None

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )
    if is_distributed:
        train_sampler = DistributedSampler(train_set, shuffle=True, seed=cfg.seed)
        dataloader = torch.utils.data.DataLoader(
            train_set, **cfg.loader, shuffle=False, drop_last=True, sampler=train_sampler
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
        )
    val_dataloader = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False
    )

    ##############################
    ##     hierarchical model   ##
    ##############################

    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    model = HierarchicalLeWM(
        jepa=jepa,
        embed_dim=cfg.wm.embed_dim,
        action_dim=effective_act_dim,
        latent_action_dim=cfg.wm.latent_action_dim,
        n_waypoints=cfg.wm.n_waypoints,
        history_size=cfg.wm.history_size,
        lambda_var=cfg.wm.lambda_var,
        lambda_kl=cfg.wm.get("lambda_kl", 0.0),
        high_depth=cfg.wm.high_depth,
        high_heads=cfg.wm.high_heads,
        high_mlp_dim=cfg.wm.high_mlp_dim,
        high_num_frames=cfg.wm.high_num_frames,
        action_enc_hidden=cfg.wm.action_enc_hidden,
        action_enc_depth=cfg.wm.action_enc_depth,
        action_enc_heads=cfg.wm.action_enc_heads,
        dropout=cfg.stage2.get("dropout", 0.0),
    )
    model = model.to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    ##########################
    ##       training       ##
    ##########################

    run_dir = Path(swm.data.utils.get_cache_dir(), cfg.get("subdir") or "")
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.yaml", "w") as f:
            OmegaConf.save(cfg, f)
    if is_distributed:
        dist.barrier()  # ensure run_dir exists before non-main ranks proceed

    py_log.info("Run directory: %s  device: %s", run_dir, device)

    wandb_run = None
    if is_main and cfg.wandb.enabled:
        wandb_run = wandb.init(**OmegaConf.to_container(cfg.wandb.config, resolve=True))
        wandb_run.config.update(OmegaConf.to_container(cfg, resolve=True))

    object_dump_callback = None
    if is_main:
        object_dump_callback = ModelObjectCallBack(
            dirpath=run_dir,
            filename=cfg.output_model_name,
            epoch_interval=cfg.stage2.get("ckpt_every_n_epochs", 1),
        )

    model = train_hierarchical_lewm(
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader if is_main else None,
        n_waypoints=cfg.wm.n_waypoints,
        lr=cfg.stage2.lr,
        n_epochs=cfg.stage2.n_epochs,
        device=device,
        freeze_encoder=cfg.stage2.freeze_encoder,
        log_every_n_steps=cfg.stage2.log_every_n_steps,
        wandb_run=wandb_run,
        ckpt_callback=object_dump_callback,
        rollout_loss=cfg.stage2.get("rollout_loss", False),
        ss_start=cfg.stage2.get("ss_start", 1.0),
        ss_end=cfg.stage2.get("ss_end", 0.25),
        weight_decay=cfg.stage2.get("weight_decay", 0.01),
        select_by=cfg.stage2.get("select_by", "tf"),
        ar_every=cfg.stage2.get("ar_every", 5),
        use_amp=cfg.stage2.get("use_amp", True),
        compile_model=cfg.stage2.get("compile", False),
    )

    if is_main:
        out_path = run_dir / f"{cfg.output_model_name}_object.ckpt"
        raw = model.module if hasattr(model, "module") else model
        torch.save(raw, out_path)
        py_log.info("Saved hierarchical model to %s", out_path)

    total_s = time.perf_counter() - t_run
    py_log.info(
        "run complete — total time: %.1f s (%.1f min)  [data+cache+train+save]",
        total_s, total_s / 60,
    )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    run()
