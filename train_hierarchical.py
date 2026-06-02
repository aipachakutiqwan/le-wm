"""Stage-2 training script for HierarchicalLeWM — PyTorch Lightning edition.

Stage 1 (JEPA) is trained by train.py.  This script loads the resulting
checkpoint, wraps it in HierarchicalLeWM, and runs the stage-2 teacher-
forcing loop that jointly trains A_ψ and P^(2).

On first run the frozen JEPA encoder is applied once to every raw frame in
the HDF5 dataset and the result is saved to disk.  Subsequent runs reuse the
cached file, so the ViT forward pass is never repeated during training.

Usage
-----
# Single GPU
python train_hierarchical.py stage1_checkpoint=<path/to/weights.pt>

# Two A100s — no torchrun needed
python train_hierarchical.py stage1_checkpoint=<path> num_gpus=2

# Different dataset
python train_hierarchical.py data=ogb stage1_checkpoint=<path>

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
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import OmegaConf, open_dict
from torchvision.transforms import v2 as T

# DataLoader workers pass tensors via /dev/shm by default. Modal containers cap
# /dev/shm small, so multi-worker loading with large batches blows it out with
# "No space left on device". file_system strategy uses regular tmpfs/RAM instead.
torch.multiprocessing.set_sharing_strategy("file_system")

from hierarchical_lewm import HierarchicalLeWM
from waypoint_sampler import sample_waypoints_fixed_stride
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
) -> None:
    """Encode all frames and write the embedding cache to disk (single-process).

    extra_device=None  → one GPU, sequential.
    extra_device="cuda:1" → two threads on two GPUs, ~2× faster.

    If out_path already exists the function returns immediately.
    """
    if out_path.exists():
        py_log.info("Reusing cached embeddings from %s", out_path)
        return

    jepa.to(device).eval()

    with h5py.File(h5_path, "r") as f:
        n_total = int(f["pixels"].shape[0])

    py_log.info(
        "Encoding %d frames on %s%s → %s",
        n_total, device,
        f" + {extra_device}" if extra_device else "",
        out_path,
    )
    t0 = time.perf_counter()

    if extra_device is not None:
        mid = n_total // 2
        jepa2 = copy.deepcopy(jepa).to(extra_device).eval()
        shards: list = [None, None]
        threads = [
            threading.Thread(
                target=_encode_worker,
                args=(jepa, h5_path, 0, mid, batch_size, device, img_size, shards, 0),
                daemon=True,
            ),
            threading.Thread(
                target=_encode_worker,
                args=(jepa2, h5_path, mid, n_total, batch_size, extra_device, img_size, shards, 1),
                daemon=True,
            ),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        all_emb = np.concatenate(shards, axis=0)
    else:
        out: list = [None]
        _encode_worker(jepa, h5_path, 0, n_total, batch_size, device, img_size, out, 0)
        all_emb = out[0]

    tmp = out_path.with_suffix(f".{os.getpid()}.tmp.npy")
    np.save(tmp, all_emb)
    tmp.rename(out_path)
    py_log.info("Embeddings saved to %s  (%.1fs)", out_path, time.perf_counter() - t0)


# ──────────────────────────────────────────────────────────────────────────────
# Lightning module
# ──────────────────────────────────────────────────────────────────────────────


class HierarchicalLeWMLit(pl.LightningModule):
    """LightningModule wrapping HierarchicalLeWM for stage-2 training.

    Lightning handles DDP, AMP (bf16-mixed), and the DataLoader DistributedSampler
    automatically.  Only A_ψ and P^(2) are optimised; the JEPA encoder is frozen
    before DDP wrapping so its parameters are excluded from all-reduce buckets.
    """

    def __init__(self, model: HierarchicalLeWM, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self._wp_idx: torch.Tensor | None = None        # reset each train epoch
        self._wp_idx_val: torch.Tensor | None = None    # reset each val epoch
        self._tf_prob: float = 1.0

        # Freeze JEPA before Lightning wraps in DDP so frozen params are
        # excluded from all-reduce buckets from the start.
        if cfg.stage2.freeze_encoder:
            for p in self.model.jepa.parameters():
                p.requires_grad_(False)

        if cfg.stage2.get("compile", False):
            self.model.action_encoder_high = torch.compile(self.model.action_encoder_high)
            self.model.high_predictor = torch.compile(self.model.high_predictor)

    # ── lifecycle hooks ────────────────────────────────────────────────────────

    def on_train_epoch_start(self) -> None:
        self._wp_idx = None   # recompute on first batch (T may change across datasets)
        if self.cfg.stage2.get("rollout_loss", False):
            n_epochs = self.cfg.stage2.n_epochs
            frac = self.current_epoch / max(1, n_epochs - 1)
            ss_start = self.cfg.stage2.get("ss_start", 1.0)
            ss_end = self.cfg.stage2.get("ss_end", 0.25)
            self._tf_prob = ss_start + (ss_end - ss_start) * frac
        else:
            self._tf_prob = 1.0

    def on_validation_epoch_start(self) -> None:
        self._wp_idx_val = None

    # ── forward passes ─────────────────────────────────────────────────────────

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        T = batch["action"].shape[1]
        if self._wp_idx is None:
            self._wp_idx = sample_waypoints_fixed_stride(
                T, N=self.cfg.wm.n_waypoints, device="cpu"
            )

        # Lightning applies bf16 autocast when precision="bf16-mixed".
        out = self.model(
            batch, self._wp_idx,
            freeze_encoder=self.cfg.stage2.freeze_encoder,
            teacher_forcing_prob=self._tf_prob,
        )

        with torch.no_grad():
            mac = out["macro_actions"]
            mac_absmean = mac.mean(dim=(0, 1)).abs().mean()
            mac_std = mac.std(dim=(0, 1)).mean()

        self.log("train/loss",         out["loss"],      on_step=True,  on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/loss_pred",    out["loss_pred"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/loss_kl",      out["loss_kl"],   on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/mac_absmean",  mac_absmean,      on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/mac_std",      mac_std,          on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/tf_prob",      self._tf_prob,    on_step=False, on_epoch=False)
        return out["loss"]

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        T = batch["action"].shape[1]
        if self._wp_idx_val is None:
            self._wp_idx_val = sample_waypoints_fixed_stride(
                T, N=self.cfg.wm.n_waypoints, device="cpu"
            )

        out = self.model.forward_high(
            batch, self._wp_idx_val,
            freeze_encoder=self.cfg.stage2.freeze_encoder,
            teacher_forcing_prob=1.0,
        )

        self.log("val/loss",      out["loss"],      on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/loss_pred", out["loss_pred"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/loss_kl",   out["loss_kl"],   on_step=False, on_epoch=True, sync_dist=True)
        return out["loss"]

    # ── optimiser ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        stage2_params = (
            list(self.model.action_encoder_high.parameters())
            + list(self.model.high_predictor.parameters())
        )
        opt = torch.optim.AdamW(
            stage2_params,
            lr=self.cfg.stage2.lr,
            weight_decay=self.cfg.stage2.get("weight_decay", 0.01),
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.cfg.stage2.n_epochs
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }


# ──────────────────────────────────────────────────────────────────────────────
# Best-model callback
# ──────────────────────────────────────────────────────────────────────────────


class _BestObjectCallback(pl.Callback):
    """Save raw HierarchicalLeWM to disk whenever val/loss improves."""

    def __init__(self, dirpath: Path, filename: str):
        self._dirpath = Path(dirpath)
        self._filename = filename
        self._best = float("inf")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: HierarchicalLeWMLit) -> None:
        if not trainer.is_global_zero:
            return
        val_loss = trainer.callback_metrics.get("val/loss")
        if val_loss is None:
            return
        val_loss = float(val_loss)
        if val_loss < self._best:
            self._best = val_loss
            path = self._dirpath / f"{self._filename}_best_object.ckpt"
            torch.save(pl_module.model, path)
            py_log.info("★ best val/loss=%.5f — saved %s", val_loss, path)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


@hydra.main(version_base=None, config_path="./config/train", config_name="hierarchical")
def run(cfg) -> None:
    t_run = time.perf_counter()

    py_log.info(
        "Hierarchical stage-2 training — data=%s  checkpoint=%s",
        cfg.data.dataset.name, cfg.stage1_checkpoint,
    )
    py_log.info("STABLEWM_HOME set to %s", os.getenv("STABLEWM_HOME"))

    with open_dict(cfg):
        cfg.data.dataset.num_steps = cfg.stage2_num_steps

    ##########################
    ##    embedding cache   ##
    ##########################
    # Run once in the main process before Lightning spawns DDP workers.
    # extra_device enables a second thread on a second GPU for ~2× speed.

    py_log.info("Loading stage-1 checkpoint from %s", cfg.stage1_checkpoint)
    jepa = torch.load(cfg.stage1_checkpoint, map_location=cfg.device, weights_only=False)
    jepa.eval()

    cache_dir = Path(cfg.get("cache_dir") or swm.data.utils.get_cache_dir())
    ckpt_stem = Path(cfg.stage1_checkpoint).stem
    emb_path = cache_dir / f"{cfg.data.dataset.name}_{ckpt_stem}_img{cfg.img_size}_emb.npy"

    _ensure_embeddings(
        jepa,
        cache_dir / f"{cfg.data.dataset.name}.h5",
        emb_path,
        cfg.img_size,
        cfg.device,
        batch_size=cfg.get("cache_batch_size", 1024),
        extra_device=cfg.get("cache_extra_device", None),
    )

    ##########################
    ##      dataset         ##
    ##########################

    dataset = _PicklableHDF5Dataset(**cfg.data.dataset, transform=None)
    emb_array = np.load(emb_path, mmap_mode="r")
    dataset._cache["emb"] = emb_array
    dataset._keys = ["emb" if k == "pixels" else k for k in dataset._keys]

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
    # shuffle=True here; Lightning replaces the sampler with DistributedSampler in DDP.
    train_loader = torch.utils.data.DataLoader(
        train_set, **loader_kwargs, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, **loader_kwargs, shuffle=False, drop_last=False
    )

    ##########################
    ##       model          ##
    ##########################

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

    lit = HierarchicalLeWMLit(model, cfg)

    ##########################
    ##    run directory     ##
    ##########################

    run_dir = Path(swm.data.utils.get_cache_dir(), cfg.get("subdir") or "")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    ##########################
    ##       logger         ##
    ##########################

    logger: pl.loggers.Logger | bool = False
    if cfg.wandb.enabled:
        wandb_cfg = OmegaConf.to_container(cfg.wandb.config, resolve=True)
        logger = WandbLogger(
            entity=wandb_cfg.get("entity"),
            project=wandb_cfg.get("project"),
            name=wandb_cfg.get("name"),
            id=wandb_cfg.get("id"),
            resume=wandb_cfg.get("resume", "allow"),
            config=OmegaConf.to_container(cfg, resolve=True),
            save_dir=str(run_dir),
        )

    ##########################
    ##      callbacks       ##
    ##########################

    callbacks = [
        ModelObjectCallBack(
            dirpath=run_dir,
            filename=cfg.output_model_name,
            epoch_interval=cfg.stage2.get("ckpt_every_n_epochs", 1),
        ),
        _BestObjectCallback(dirpath=run_dir, filename=cfg.output_model_name),
    ]

    ##########################
    ##       trainer        ##
    ##########################

    num_gpus = cfg.get("num_gpus", 1)
    strategy: str | DDPStrategy = (
        DDPStrategy(find_unused_parameters=True) if num_gpus > 1 else "auto"
    )
    accelerator = "gpu" if cfg.device.startswith("cuda") else cfg.device

    trainer = pl.Trainer(
        devices=num_gpus,
        accelerator=accelerator,
        strategy=strategy,
        precision="bf16-mixed" if cfg.stage2.get("use_amp", True) else "32-true",
        max_epochs=cfg.stage2.n_epochs,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.stage2.log_every_n_steps,
        default_root_dir=str(run_dir),
        enable_progress_bar=True,
    )

    trainer.fit(lit, train_loader, val_loader)

    ##########################
    ##      save final      ##
    ##########################

    if trainer.is_global_zero:
        out_path = run_dir / f"{cfg.output_model_name}_object.ckpt"
        torch.save(model, out_path)
        py_log.info("Saved final model to %s", out_path)

        total_s = time.perf_counter() - t_run
        py_log.info(
            "run complete — total time: %.1f s (%.1f min)",
            total_s, total_s / 60,
        )


if __name__ == "__main__":
    run()
