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
    rank: int = 0,
    world_size: int = 1,
) -> None:
    """Encode this rank's frame shard and write it to disk.

    When world_size > 1 each rank encodes rows
    [rank*n//world_size, (rank+1)*n//world_size) and writes a shard file.
    The caller is responsible for the two dist.barrier() calls and the
    rank-0 merge step.  When world_size == 1 this writes directly to
    out_path (via a tmp-rename).  If out_path already exists (merged cache
    from a previous run) the function returns immediately on all ranks.
    """
    if out_path.exists():
        if rank == 0:
            py_log.info("Reusing cached embeddings from %s", out_path)
        return

    if world_size == 1:
        shard_path = out_path          # single rank: write straight to final path
    else:
        shard_path = out_path.with_name(f"{out_path.stem}.shard{rank}.npy")
        if shard_path.exists():
            py_log.info("[rank %d] Shard already on disk, skipping encode", rank)
            return

    py_log.info("[rank %d/%d] Encoding shard from %s …", rank, world_size, h5_path)
    jepa.to(device).eval()
    stats = spt.data.dataset_stats.ImageNet
    transform = T.Compose([
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=stats["mean"], std=stats["std"]),
        T.Resize(size=img_size, antialias=True),
    ])

    t0 = time.perf_counter()
    # h5py SWMR mode allows concurrent readers on the same file — safe for DDP.
    with h5py.File(h5_path, "r", swmr=True) as f:
        n_total = int(f["pixels"].shape[0])
        row_start = rank * n_total // world_size
        row_end = (rank + 1) * n_total // world_size
        n_shard = row_end - row_start
        py_log.info("[rank %d] rows %d–%d (%d frames)", rank, row_start, row_end, n_shard)
        all_emb = None

        for s in range(row_start, row_end, batch_size):
            e = min(s + batch_size, row_end)
            raw = f["pixels"][s:e]                              # (B, H, W, C) uint8
            frames = torch.from_numpy(raw).permute(0, 3, 1, 2) # (B, C, H, W)
            frames = transform(frames).unsqueeze(1).to(device)  # (B, 1, C, H, W)
            with torch.no_grad():
                emb = jepa.encode({"pixels": frames})["emb"][:, 0]  # (B, D)
            if all_emb is None:
                all_emb = np.zeros((n_shard, emb.shape[-1]), dtype=np.float32)
            local_s = s - row_start
            local_e = e - row_start
            all_emb[local_s:local_e] = emb.float().cpu().numpy()

            if (local_s // batch_size) % 20 == 0 or e == row_end:
                py_log.info(
                    "[rank %d]  %.1f%%  (%d/%d)  %.1fs",
                    rank, 100.0 * local_e / n_shard, local_e, n_shard,
                    time.perf_counter() - t0,
                )

    tmp = shard_path.with_suffix(f".{os.getpid()}.tmp.npy")
    np.save(tmp, all_emb)
    tmp.rename(shard_path)
    py_log.info("[rank %d] Shard saved to %s  (%.1fs)", rank, shard_path, time.perf_counter() - t0)


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

    _rank = dist.get_rank() if is_distributed else 0
    _world = dist.get_world_size() if is_distributed else 1

    # All ranks encode their own shard in parallel (~2× faster on 2 A100s).
    _ensure_embeddings(
        jepa, cache_dir / f"{cfg.data.dataset.name}.h5", emb_path,
        cfg.img_size, device,
        batch_size=cfg.get("cache_batch_size", 1024),
        rank=_rank,
        world_size=_world,
    )

    if is_distributed:
        dist.barrier()  # wait for every rank's shard to land on disk
        if is_main and not emb_path.exists():
            shard_files = [
                emb_path.with_name(f"{emb_path.stem}.shard{r}.npy")
                for r in range(_world)
            ]
            py_log.info("Merging %d shards into %s", _world, emb_path)
            merged = np.concatenate([np.load(p) for p in shard_files], axis=0)
            tmp = emb_path.with_suffix(f".{os.getpid()}.tmp.npy")
            np.save(tmp, merged)
            tmp.rename(emb_path)
            for p in shard_files:
                p.unlink(missing_ok=True)
            py_log.info("Merge complete — %d frames total", merged.shape[0])
        dist.barrier()  # wait for rank-0 merge before everyone loads

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
