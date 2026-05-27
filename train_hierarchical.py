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
from functools import partial
from pathlib import Path

py_log = logging.getLogger(__name__)

import hydra
import wandb
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf, open_dict

# DataLoader workers pass tensors via /dev/shm by default. Modal containers cap
# /dev/shm small, so multi-worker loading with large batches blows it out with
# "No space left on device". file_system strategy uses regular tmpfs/RAM instead.
torch.multiprocessing.set_sharing_strategy("file_system")

from hierarchical_lewm import HierarchicalLeWM, train_hierarchical_lewm
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


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
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )
    dataloader = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False
    )

    ##############################
    ##       model              ##
    ##############################

    py_log.info("STABLEWM_HOME set to %s", os.getenv('STABLEWM_HOME'))

    py_log.info("Loading stage-1 checkpoint from %s", cfg.stage1_checkpoint)
    jepa = torch.load(cfg.stage1_checkpoint, map_location=cfg.device, weights_only=False)
    jepa.eval()

    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    model = HierarchicalLeWM(
        jepa=jepa,
        embed_dim=cfg.wm.embed_dim,
        action_dim=effective_act_dim,
        latent_action_dim=cfg.wm.latent_action_dim,
        n_waypoints=cfg.wm.n_waypoints,
        history_size=cfg.wm.history_size,
        lambda_var=cfg.wm.lambda_var,
        high_depth=cfg.wm.high_depth,
        high_heads=cfg.wm.high_heads,
        high_mlp_dim=cfg.wm.high_mlp_dim,
        high_num_frames=cfg.wm.high_num_frames,
        action_enc_hidden=cfg.wm.action_enc_hidden,
        action_enc_depth=cfg.wm.action_enc_depth,
        action_enc_heads=cfg.wm.action_enc_heads,
        dropout=cfg.stage2.get("dropout", 0.0),
    )

    ##########################
    ##       training       ##
    ##########################

    run_dir = Path(swm.data.utils.get_cache_dir(), cfg.get("subdir") or "")
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    device = cfg.device
    py_log.info("Run directory: %s  device: %s", run_dir, device)

    wandb_run = None
    if cfg.wandb.enabled:
        wandb_run = wandb.init(**OmegaConf.to_container(cfg.wandb.config, resolve=True))
        wandb_run.config.update(OmegaConf.to_container(cfg, resolve=True))

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir,
        filename=cfg.output_model_name,
        epoch_interval=cfg.stage2.get("ckpt_every_n_epochs", 1),
    )

    model = train_hierarchical_lewm(
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
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
    )

    out_path = run_dir / f"{cfg.output_model_name}_object.ckpt"
    torch.save(model, out_path)
    py_log.info("Saved hierarchical model to %s", out_path)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    run()
