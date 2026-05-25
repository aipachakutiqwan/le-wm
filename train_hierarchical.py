"""Stage-2 training script for HierarchicalLeWM.

Stage 1 (JEPA) is trained by train.py.  This script loads the resulting
checkpoint, wraps it in HierarchicalLeWM, and runs the stage-2 teacher-
forcing loop that jointly trains A_ψ and P^(2).

Usage
-----
# TwoRoom (default data)
python train_hierarchical.py stage1_checkpoint=<path/to/lewm_epoch_100_object.ckpt>

# Different dataset
python train_hierarchical.py data=pusht stage1_checkpoint=<path>

# Quick smoke-test
python train_hierarchical.py stage1_checkpoint=<path> \\
    stage2.n_epochs=2 loader.batch_size=8 wandb.enabled=False
"""

import os
import logging
from pathlib import Path

py_log = logging.getLogger(__name__)

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from hierarchical_lewm import HierarchicalLeWM, sample_waypoints
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


# ──────────────────────────────────────────────────────────────────────────────
# Lightning module
# ──────────────────────────────────────────────────────────────────────────────


class HierarchicalStage2Module(pl.LightningModule):
    """Lightning wrapper for stage-2 training of HierarchicalLeWM.

    Only A_ψ and P^(2) are optimised; E and P^(1) are frozen.
    Mirrors the spt.Module pattern used by train.py so multi-GPU DDP is
    handled transparently by pl.Trainer.
    """

    def __init__(self, model: HierarchicalLeWM, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg

    def on_fit_start(self):
        if self.cfg.stage2.freeze_encoder:
            for p in self.model.jepa.parameters():
                p.requires_grad_(False)

    def _step(self, batch):
        T = batch["pixels"].shape[1]
        wp_idx = sample_waypoints(T, N=self.cfg.wm.n_waypoints, device=self.device)
        return self.model.forward_high(
            batch, wp_idx, freeze_encoder=self.cfg.stage2.freeze_encoder
        )

    def training_step(self, batch, batch_idx):
        out = self._step(batch)
        self.log("stage2/train_loss", out["loss"], on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("stage2/train_loss_tf", out["loss_tf"], on_step=True, on_epoch=True, sync_dist=True)
        if self.cfg.wm.lambda_sigreg > 0.0:
            self.log("stage2/train_loss_reg", out["loss_reg"], on_step=True, on_epoch=True, sync_dist=True)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self._step(batch)
        self.log("stage2/val_loss", out["loss"], on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("stage2/val_loss_tf", out["loss_tf"], on_step=False, on_epoch=True, sync_dist=True)
        if self.cfg.wm.lambda_sigreg > 0.0:
            self.log("stage2/val_loss_reg", out["loss_reg"], on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        params = (
            list(self.model.action_encoder_high.parameters())
            + list(self.model.high_predictor.parameters())
        )
        return torch.optim.AdamW(params, lr=self.cfg.stage2.lr)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


@hydra.main(version_base=None, config_path="./config/train", config_name="hierarchical")
def run(cfg):
    py_log.info(
        "Hierarchical stage-2 training — data=%s setup=%s wandb=%s checkpoint=%s",
        cfg.data.dataset.name,
        cfg.get("setup", "default"),
        cfg.wandb.enabled,
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
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )
    dataloader = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False
    )
    py_log.info("Dataset split — train: %d  val: %d", len(train_set), len(val_set))

    ##############################
    ##       model              ##
    ##############################

    py_log.info("STABLEWM_HOME set to %s", os.getenv('STABLEWM_HOME'))
    py_log.info("Loading stage-1 checkpoint from %s", cfg.stage1_checkpoint)
    # map_location="cpu" is required for DDP: each rank loads to CPU first,
    # then Lightning moves parameters to the correct device.
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
        lambda_sigreg=cfg.wm.lambda_sigreg,
        high_depth=cfg.wm.high_depth,
        high_heads=cfg.wm.high_heads,
        high_mlp_dim=cfg.wm.high_mlp_dim,
        high_num_frames=cfg.wm.high_num_frames,
        action_enc_hidden=cfg.wm.action_enc_hidden,
        action_enc_depth=cfg.wm.action_enc_depth,
        action_enc_heads=cfg.wm.action_enc_heads,
    )

    total_params = sum(p.numel() for p in model.parameters())
    stage2_params = sum(
        p.numel() for p in list(model.action_encoder_high.parameters()) + list(model.high_predictor.parameters())
    )
    py_log.info(
        "Model — total params: %d  stage-2 trainable: %d  frozen: %d  "
        "embed_dim: %d  latent_action_dim: %d  n_waypoints: %d",
        total_params, stage2_params, total_params - stage2_params,
        cfg.wm.embed_dim, cfg.wm.latent_action_dim, cfg.wm.n_waypoints,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    # Re-enable submodule logger disabled by Hydra's hydra_logging phase.
    logging.getLogger("hierarchical_lewm").disabled = False
    py_log.info("Run Directory: %s", run_dir)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    lm = HierarchicalStage2Module(model, cfg)
    data_module = spt.data.DataModule(train=dataloader, val=val_dataloader)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )
    trainer.fit(lm, datamodule=data_module)

    if trainer.is_global_zero:
        out_path = run_dir / f"{cfg.output_model_name}_object.ckpt"
        torch.save(lm.model, out_path)
        py_log.info("Training complete — checkpoint saved to %s", out_path)


if __name__ == "__main__":
    run()
