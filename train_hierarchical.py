"""Stage-2 training script for HierarchicalLeWM.

Stage 1 (JEPA) is trained by train.py.  This script loads the resulting
checkpoint, wraps it in HierarchicalLeWMModule, and runs the stage-2
teacher-forcing loop using PyTorch Lightning — supports 1 GPU or multi-GPU
DDP transparently via ``trainer.devices: auto`` in the config.

On first run the frozen JEPA encoder is applied once to every raw frame in the
HDF5 dataset and the result is saved to disk (``{dataset}_{ckpt}_img{N}_emb.npy``).
Subsequent runs with the same stage-1 checkpoint reuse the cached file, so
the ViT forward pass is never repeated during training.

Usage
-----
# TwoRoom (default data), single or multi-GPU depending on visible devices
python train_hierarchical.py stage1_checkpoint=<path/to/lewm_epoch_100_object.ckpt>

# Different dataset
python train_hierarchical.py data=pusht stage1_checkpoint=<path>

# Quick smoke-test (CPU / MPS — hardware is auto-detected)
python train_hierarchical.py stage1_checkpoint=<path> \\
    trainer.max_epochs=2 loader.batch_size=8 loader.num_workers=0 \\
    wandb.enabled=False stage2.compile=false
"""

import logging
import os
from pathlib import Path
import time

import h5py
import numpy as np
import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from torchvision.transforms import v2 as T

from hierarchical_lewm import HierarchicalLeWM, HierarchicalLeWMModule
from jepa import JEPA
from module import ARPredictor, Embedder, MLP
from utils import get_column_normalizer

py_log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# JEPA builder  (used when loading from weights.pt state-dict format)
# ──────────────────────────────────────────────────────────────────────────────


def _build_jepa(cfg, effective_act_dim: int) -> JEPA:
    """Reconstruct JEPA architecture from config and return an un-initialised model.

    The caller must load a state dict into the returned model.  The architecture
    must exactly match the stage-1 run that produced the weights.pt file.
    """
    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )
    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    projector = MLP(
        input_dim=hidden_dim, output_dim=embed_dim,
        hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d,
    )
    pred_proj = MLP(
        input_dim=hidden_dim, output_dim=embed_dim,
        hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d,
    )
    return JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Embedding cache
# ──────────────────────────────────────────────────────────────────────────────


def _ensure_embeddings(
    jepa: torch.nn.Module,
    h5_path: Path,
    out_path: Path,
    img_size: int,
    device: str,
    batch_size: int = 256,
    local_rank: int = 0,
) -> np.ndarray:
    """Return cached embeddings, computing and saving them first if needed.

    The cache file is keyed to both the dataset and the stage-1 checkpoint stem,
    so switching checkpoints never produces stale embeddings.

    In multi-GPU runs only rank 0 computes; other ranks wait for the file.
    """
    if out_path.exists():
        py_log.info("Reusing cached embeddings from %s", out_path)
        return np.load(out_path, mmap_mode="r")

    if local_rank != 0:
        py_log.info("Rank %d: waiting for rank 0 to compute embeddings …", local_rank)
        while not out_path.exists():
            time.sleep(2)
        return np.load(out_path, mmap_mode="r")

    py_log.info("Embedding cache not found — computing from %s", h5_path)
    jepa.to(device)  # model was loaded to CPU; move to compute device
    stats = spt.data.dataset_stats.ImageNet
    transform = T.Compose([
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=stats["mean"], std=stats["std"]),
        T.Resize(size=img_size, antialias=True),
    ])

    with h5py.File(h5_path, "r", swmr=True) as f:
        n_total = int(f["pixels"].shape[0])
        py_log.info("Raw frames to encode: %d", n_total)

        dummy = torch.from_numpy(f["pixels"][:1]).permute(0, 3, 1, 2)
        dummy = transform(dummy).unsqueeze(1).to(device)
        with torch.no_grad():
            embed_dim = jepa.encode({"pixels": dummy})["emb"].shape[-1]
        py_log.info("embed_dim: %d", embed_dim)

        all_emb = np.zeros((n_total, embed_dim), dtype=np.float32)
        t0 = time.perf_counter()

        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            raw = f["pixels"][start:end]                           # (B, H, W, C) uint8
            frames = torch.from_numpy(raw).permute(0, 3, 1, 2)    # (B, C, H, W)
            frames = transform(frames).unsqueeze(1).to(device)    # (B, 1, C, H, W)
            with torch.no_grad():
                emb = jepa.encode({"pixels": frames})["emb"]      # (B, 1, D)
            all_emb[start:end] = emb[:, 0].cpu().numpy()

            if (start // batch_size) % 20 == 0 or end == n_total:
                py_log.info(
                    "  %.1f%%  (%d/%d frames)  %.1fs",
                    100.0 * end / n_total, end, n_total, time.perf_counter() - t0,
                )

    # Atomic write: save to a per-process temp file then rename so concurrent
    # writers (shouldn't happen, but guard anyway) never corrupt the output.
    tmp = out_path.with_suffix(f".{os.getpid()}.tmp.npy")
    np.save(tmp, all_emb)
    tmp.rename(out_path)
    py_log.info("Embeddings saved to %s  (%.1fs)", out_path, time.perf_counter() - t0)
    return np.load(out_path, mmap_mode="r")


# ──────────────────────────────────────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────────────────────────────────────


class EpochTimer(Callback):
    """Log per-epoch wall time and total training time (rank 0 only)."""

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._t_train = time.perf_counter()
        self._t_epoch = self._t_train

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._t_epoch = time.perf_counter()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        total = trainer.max_epochs
        elapsed = time.perf_counter() - self._t_epoch
        steps_per_sec = trainer.num_training_batches / elapsed
        eta_s = (total - epoch) * elapsed
        py_log.info(
            "epoch %d/%d — %.1f s  (%.2f steps/s  ETA %.1f min)",
            epoch, total, elapsed, steps_per_sec, eta_s / 60,
        )

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._t_val = time.perf_counter()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        elapsed = time.perf_counter() - self._t_val
        py_log.info("val epoch %d — %.1f s", trainer.current_epoch + 1, elapsed)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        total = time.perf_counter() - self._t_train
        py_log.info("training complete — total time: %.1f s (%.1f min)", total, total / 60)


def _save_model(model, path: Path) -> None:
    """Save model with torch.compile wrappers stripped (portable checkpoint).

    Temporarily swaps OptimizedModules back to their originals, saves, then
    restores so training continues with the compiled versions.
    """
    compiled_ae = model.action_encoder_high
    compiled_hp = model.high_predictor
    model.action_encoder_high = getattr(compiled_ae, '_orig_mod', compiled_ae)
    model.high_predictor = getattr(compiled_hp, '_orig_mod', compiled_hp)
    torch.save(model, path)
    model.action_encoder_high = compiled_ae
    model.high_predictor = compiled_hp


class EpochCheckpoint(Callback):
    """Save module.model as a plain torch object after every epoch (rank 0 only)."""

    def __init__(self, run_dir: Path, model_name: str):
        self.run_dir = run_dir
        self.model_name = model_name

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        path = self.run_dir / f"{self.model_name}_epoch_{epoch}_object.ckpt"
        _save_model(pl_module.model, path)
        py_log.info("Saved epoch-%d checkpoint to %s", epoch, path)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


@hydra.main(version_base=None, config_path="./config/train", config_name="hierarchical")
def run(cfg):
    torch.set_float32_matmul_precision("high")
    t_run = time.perf_counter()
    py_log.info(
        "Hierarchical stage-2 training — data=%s checkpoint=%s",
        cfg.data.dataset.name,
        cfg.stage1_checkpoint,
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

    py_log.info("Loading stage-1 checkpoint from %s", cfg.stage1_checkpoint)
    ckpt_path = Path(cfg.stage1_checkpoint)
    if ckpt_path.suffix == ".pt":
        # weights.pt format: state dict only — rebuild architecture from config.
        effective_act_dim_jepa = cfg.data.dataset.frameskip * dataset.get_dim("action")
        jepa = _build_jepa(cfg, effective_act_dim_jepa)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        jepa.load_state_dict(state)
        py_log.info("Loaded state dict from %s", ckpt_path)
    else:
        # Legacy object format (.ckpt) — full pickle load.
        jepa = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    jepa.eval()

    ##############################
    ##   embedding cache        ##
    ##############################

    cache_dir = Path(cfg.get("cache_dir") or swm.data.utils.get_cache_dir())
    ckpt_stem = Path(cfg.stage1_checkpoint).stem
    emb_path = cache_dir / f"{cfg.data.dataset.name}_{ckpt_stem}_img{cfg.img_size}_emb.npy"

    emb_device = "cuda" if torch.cuda.is_available() else "cpu"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    emb_array = _ensure_embeddings(
        jepa, cache_dir / f"{cfg.data.dataset.name}.h5", emb_path,
        cfg.img_size, emb_device, local_rank=local_rank,
    )

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

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )
    py_log.info("train=%d  val=%d samples", len(train_set), len(val_set))
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
        lambda_sigreg=cfg.wm.get("lambda_sigreg", 0.0),
        lambda_mac_reg=cfg.wm.get("lambda_mac_reg", 0.0),
        high_depth=cfg.wm.high_depth,
        high_heads=cfg.wm.high_heads,
        high_mlp_dim=cfg.wm.high_mlp_dim,
        high_num_frames=cfg.wm.high_num_frames,
        high_dropout=cfg.wm.get("high_dropout", 0.0),
        action_enc_hidden=cfg.wm.action_enc_hidden,
        action_enc_depth=cfg.wm.action_enc_depth,
        action_enc_heads=cfg.wm.action_enc_heads,
        action_enc_dropout=cfg.wm.get("action_enc_dropout", 0.0),
    )

    module = HierarchicalLeWMModule(
        model=model,
        n_waypoints=cfg.wm.n_waypoints,
        lr=cfg.stage2.lr,
        weight_decay=cfg.stage2.get("weight_decay", 0.0),
        freeze_encoder=cfg.stage2.freeze_encoder,
        compile_model=cfg.stage2.get("compile", True),
        ss_max_prob=cfg.stage2.get("ss_max_prob", 0.0),
        ss_ramp_epochs=cfg.stage2.get("ss_ramp_epochs", 30),
    )

    ##########################
    ##       logging        ##
    ##########################

    run_dir = Path(swm.data.utils.get_cache_dir(), cfg.get("subdir") or "")
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**OmegaConf.to_container(cfg.wandb.config, resolve=True))
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    ##########################
    ##       training       ##
    ##########################

    # Patch trainer config for the available hardware before passing to Lightning.
    # ddp_find_unused_parameters_false and bf16-mixed both require multi-GPU CUDA;
    # strip them automatically so the same config works on CPU and MPS too.
    with open_dict(cfg):
        cuda_ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
        accel = cfg.trainer.get("accelerator", "auto")
        on_cpu_or_mps = (not cuda_ok) or (accel in ("cpu", "mps"))
        single_device = cuda_ok and torch.cuda.device_count() == 1

        if on_cpu_or_mps or single_device:
            cfg.trainer.pop("strategy", None)

        if on_cpu_or_mps:
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.trainer.precision = "32-true"

        # persistent_workers=True requires num_workers > 0.
        if cfg.loader.get("num_workers", 0) == 0:
            cfg.loader.persistent_workers = False

    py_log.info(
        "trainer: accelerator=%s  devices=%s  precision=%s  strategy=%s",
        cfg.trainer.get("accelerator"), cfg.trainer.get("devices"),
        cfg.trainer.get("precision"), cfg.trainer.get("strategy", "none"),
    )

    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer, resolve=True),
        logger=logger,
        enable_checkpointing=False,
        callbacks=[EpochTimer(), EpochCheckpoint(run_dir, cfg.output_model_name)],
    )

    py_log.info("Run directory: %s", run_dir)
    trainer.fit(module, dataloader, val_dataloaders=val_dataloader)

    ##########################
    ##        save          ##
    ##########################

    if trainer.is_global_zero:
        out_path = run_dir / f"{cfg.output_model_name}_object.ckpt"
        _save_model(module.model, out_path)
        py_log.info("Saved hierarchical model to %s", out_path)
        total_s = time.perf_counter() - t_run
        py_log.info("run complete — total time: %.1f s (%.1f min)", total_s, total_s / 60)


if __name__ == "__main__":
    run()
