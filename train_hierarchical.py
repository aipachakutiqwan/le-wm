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

import copy
import os
import logging
import threading
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
# Picklable HDF5Dataset
# ──────────────────────────────────────────────────────────────────────────────


class _PicklableHDF5Dataset(swm.data.HDF5Dataset):
    """HDF5Dataset with proper pickle support for DataLoader multi-worker loading.

    HDF5Dataset._open() stores an open h5py.File in self.h5_file.  Open file
    handles cannot be pickled, which prevents DataLoader from spawning worker
    processes.  __getstate__ closes and drops the handle before serialisation;
    __setstate__ restores the rest of the instance so that _open() can reopen
    the file lazily on the first __getitem__ call inside each worker.
    """

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        if state.get("h5_file") is not None:
            state["h5_file"].close()
            state["h5_file"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        # h5_file is None; _open() will reopen on next _load_slice call.


# ──────────────────────────────────────────────────────────────────────────────
# Embedding cache
# ──────────────────────────────────────────────────────────────────────────────


def _encode_worker(
    jepa: torch.nn.Module,
    h5_path: Path,
    row_start: int,
    row_end: int,
    batch_size: int,
    device: str,
    img_size: int,
    out: list,
    idx: int,
) -> None:
    """Thread worker: encode rows [row_start, row_end) on `device`; store array in out[idx]."""
    stats = spt.data.dataset_stats.ImageNet
    transform = T.Compose([
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=stats["mean"], std=stats["std"]),
        T.Resize(size=img_size, antialias=True),
    ])
    n = row_end - row_start
    arr = None
    t0 = time.perf_counter()
    with h5py.File(h5_path, "r") as f:
        for s in range(row_start, row_end, batch_size):
            e = min(s + batch_size, row_end)
            raw = f["pixels"][s:e]
            frames = torch.from_numpy(raw).permute(0, 3, 1, 2)
            frames = transform(frames).unsqueeze(1).to(device)
            with torch.no_grad():
                emb = jepa.encode({"pixels": frames})["emb"][:, 0]
            if arr is None:
                arr = np.zeros((n, emb.shape[-1]), dtype=np.float32)
            arr[s - row_start : e - row_start] = emb.float().cpu().numpy()
            done = e - row_start
            if (done // batch_size) % 20 == 0 or e == row_end:
                py_log.info(
                    "[%s]  %.1f%%  (%d/%d)  %.1fs",
                    device, 100.0 * done / n, done, n, time.perf_counter() - t0,
                )
    out[idx] = arr


def _ensure_embeddings(
    jepa: torch.nn.Module,
    h5_path: Path,
    out_path: Path,
    img_size: int,
    device: str,
    batch_size: int = 1024,
    extra_device: str | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> None:
    """Encode this rank's frame shard and write it to disk.

    Single-process (world_size == 1):
      - extra_device=None  → one GPU, sequential
      - extra_device="cuda:1" → two threads on two GPUs, ~2× faster

    DDP (world_size > 1, launched via torchrun):
      - each rank independently encodes its shard; extra_device is ignored
      - caller handles the two dist.barrier() + rank-0 merge

    If out_path already exists (merged cache from a previous run) all ranks
    return immediately.
    """
    if out_path.exists():
        if rank == 0:
            py_log.info("Reusing cached embeddings from %s", out_path)
        return

    if world_size == 1:
        shard_path = out_path
    else:
        shard_path = out_path.with_name(f"{out_path.stem}.shard{rank}.npy")
        if shard_path.exists():
            py_log.info("[rank %d] Shard already on disk, skipping encode", rank)
            return

    jepa.to(device).eval()

    with h5py.File(h5_path, "r") as f:
        n_total = int(f["pixels"].shape[0])

    row_start = rank * n_total // world_size
    row_end = (rank + 1) * n_total // world_size
    n_shard = row_end - row_start

    py_log.info(
        "[rank %d/%d] Encoding rows %d–%d (%d frames) on %s%s",
        rank, world_size, row_start, row_end, n_shard, device,
        f" + {extra_device}" if extra_device and world_size == 1 else "",
    )

    t0 = time.perf_counter()

    if extra_device is not None and world_size == 1:
        # Two-GPU threading: split the shard evenly between primary and extra device.
        mid = (row_start + row_end) // 2
        jepa2 = copy.deepcopy(jepa).to(extra_device).eval()
        shards: list = [None, None]
        threads = [
            threading.Thread(
                target=_encode_worker,
                args=(jepa, h5_path, row_start, mid, batch_size, device, img_size, shards, 0),
                daemon=True,
            ),
            threading.Thread(
                target=_encode_worker,
                args=(jepa2, h5_path, mid, row_end, batch_size, extra_device, img_size, shards, 1),
                daemon=True,
            ),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        all_emb = np.concatenate(shards, axis=0)
    else:
        # Single-GPU path (DDP shard or explicit single-GPU run).
        out: list = [None]
        _encode_worker(jepa, h5_path, row_start, row_end, batch_size, device, img_size, out, 0)
        all_emb = out[0]

    tmp = shard_path.with_suffix(f".{os.getpid()}.tmp.npy")
    np.save(tmp, all_emb)
    tmp.rename(shard_path)
    py_log.info(
        "[rank %d] Embeddings saved to %s  (%.1fs)",
        rank, shard_path, time.perf_counter() - t0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Training worker (called directly for 1 GPU, or via mp.spawn for N GPUs)
# ──────────────────────────────────────────────────────────────────────────────


def _do_train(rank: int, world_size: int, cfg) -> None:
    t_run = time.perf_counter()

    is_distributed = world_size > 1
    is_main = rank == 0
    device = f"cuda:{rank}" if is_distributed else cfg.device

    py_log.info(
        "Hierarchical stage-2 training — data=%s checkpoint=%s  rank=%d/%d",
        cfg.data.dataset.name, cfg.stage1_checkpoint, rank, world_size,
    )

    #########################
    ##       dataset       ##
    #########################

    with open_dict(cfg):
        cfg.data.dataset.num_steps = cfg.stage2_num_steps

    dataset = _PicklableHDF5Dataset(**cfg.data.dataset, transform=None)

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

    # Each rank encodes its own shard; in single-process mode cache_extra_device
    # enables a second thread on a second GPU.
    _ensure_embeddings(
        jepa, cache_dir / f"{cfg.data.dataset.name}.h5", emb_path,
        cfg.img_size, device,
        batch_size=cfg.get("cache_batch_size", 1024),
        extra_device=cfg.get("cache_extra_device", None),
        rank=rank,
        world_size=world_size,
    )

    if is_distributed:
        dist.barrier()  # wait for every rank's shard to land on disk
        if is_main and not emb_path.exists():
            shard_files = [
                emb_path.with_name(f"{emb_path.stem}.shard{r}.npy")
                for r in range(world_size)
            ]
            py_log.info("Merging %d shards into %s", world_size, emb_path)
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

    loader_kwargs = OmegaConf.to_container(cfg.loader)

    if is_distributed:
        train_sampler = DistributedSampler(train_set, shuffle=True, seed=cfg.seed)
        dataloader = torch.utils.data.DataLoader(
            train_set, **loader_kwargs, shuffle=False, drop_last=True, sampler=train_sampler
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            train_set, **loader_kwargs, shuffle=True, drop_last=True, generator=rnd_gen
        )
    val_dataloader = torch.utils.data.DataLoader(
        val_set, **loader_kwargs, shuffle=False, drop_last=False
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
    if cfg.stage2.freeze_encoder:
        # Freeze before DDP so the frozen params are excluded from all-reduce buckets.
        # train_hierarchical_lewm repeats this call, but it becomes a no-op.
        for p in model.jepa.parameters():
            p.requires_grad_(False)
    if is_distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

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


def _ddp_worker(rank: int, world_size: int, cfg_dict: dict) -> None:
    """mp.spawn entry point: initialise DDP then run training."""
    torch.multiprocessing.set_sharing_strategy("file_system")
    cfg = OmegaConf.create(cfg_dict)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(cfg.get("ddp_port", 29500)))
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    try:
        _do_train(rank=rank, world_size=world_size, cfg=cfg)
    finally:
        dist.destroy_process_group()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


@hydra.main(version_base=None, config_path="./config/train", config_name="hierarchical")
def run(cfg) -> None:
    num_gpus = cfg.get("num_gpus", 1)
    if num_gpus > 1:
        # Spawn one process per GPU — no torchrun needed.
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        torch.multiprocessing.spawn(
            _ddp_worker,
            args=(num_gpus, cfg_dict),
            nprocs=num_gpus,
            join=True,
        )
    else:
        _do_train(rank=0, world_size=1, cfg=cfg)


if __name__ == "__main__":
    run()
