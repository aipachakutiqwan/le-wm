"""Stage-2 training script for HierarchicalLeWM.

Stage 1 (JEPA) is trained by train.py.  This script loads the resulting
checkpoint, wraps it in HierarchicalLeWMModule, and runs the stage-2
teacher-forcing loop using PyTorch Lightning — supports 1 GPU or multi-GPU
DDP transparently via ``trainer.devices: auto`` in the config.

Usage
-----
# TwoRoom (default data), single or multi-GPU depending on visible devices
python train_hierarchical.py stage1_checkpoint=<path/to/lewm_epoch_100_object.ckpt>

# Different dataset
python train_hierarchical.py data=pusht stage1_checkpoint=<path>

# Quick smoke-test (CPU)
python train_hierarchical.py stage1_checkpoint=<path> \\
    trainer.max_epochs=2 loader.batch_size=8 \\
    trainer.accelerator=cpu trainer.devices=1 wandb.enabled=False
"""

import logging
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from hierarchical_lewm import HierarchicalLeWM, HierarchicalLeWMModule
from utils import get_column_normalizer, get_img_preprocessor

py_log = logging.getLogger(__name__)


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
        torch.save(pl_module.model, path)
        py_log.info("Saved epoch-%d checkpoint to %s", epoch, path)


@hydra.main(version_base=None, config_path="./config/train", config_name="hierarchical")
def run(cfg):
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
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]

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
    train_set, _ = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )
    dataloader = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )

    ##############################
    ##       model              ##
    ##############################

    py_log.info("Loading stage-1 checkpoint from %s", cfg.stage1_checkpoint)
    jepa = torch.load(cfg.stage1_checkpoint, map_location="cpu", weights_only=False)
    jepa.eval()

    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    model = HierarchicalLeWM(
        jepa=jepa,
        embed_dim=cfg.wm.embed_dim,
        action_dim=effective_act_dim,
        latent_action_dim=cfg.wm.latent_action_dim,
        n_waypoints=cfg.wm.n_waypoints,
        history_size=cfg.wm.history_size,
        high_depth=cfg.wm.high_depth,
        high_heads=cfg.wm.high_heads,
        high_mlp_dim=cfg.wm.high_mlp_dim,
        high_num_frames=cfg.wm.high_num_frames,
        action_enc_hidden=cfg.wm.action_enc_hidden,
        action_enc_depth=cfg.wm.action_enc_depth,
        action_enc_heads=cfg.wm.action_enc_heads,
    )

    module = HierarchicalLeWMModule(
        model=model,
        n_waypoints=cfg.wm.n_waypoints,
        lr=cfg.stage2.lr,
        freeze_encoder=cfg.stage2.freeze_encoder,
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

    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer, resolve=True),
        logger=logger,
        enable_checkpointing=False,
        callbacks=[EpochCheckpoint(run_dir, cfg.output_model_name)],
    )

    py_log.info("Run directory: %s", run_dir)
    trainer.fit(module, dataloader)

    ##########################
    ##        save          ##
    ##########################

    if trainer.is_global_zero:
        out_path = run_dir / f"{cfg.output_model_name}_object.ckpt"
        torch.save(module.model, out_path)
        py_log.info("Saved hierarchical model to %s", out_path)


if __name__ == "__main__":
    run()
