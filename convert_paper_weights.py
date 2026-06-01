#!/usr/bin/env python3
"""Convert paper weights (weights.pt + config.json) into a JEPA _object.ckpt.

The HuggingFace model repos store weights as a raw state dict (weights.pt) plus
an architecture config (config.json). The eval pipeline expects a pickled JEPA
instance (*_object.ckpt). This script bridges the gap.

config.json references stable_worldmodel.wm.lewm.module.* which does not exist
in the installed package — the actual classes live in the repo's module.py.
We instantiate them directly from the JSON params.

Output: baseline_paper/<env>/lewm_paper_object.ckpt
        → usable as policy=/path/to/baseline_paper/<env>

Usage:
    python convert_paper_weights.py                   # all envs
    python convert_paper_weights.py --only tworooms   # single env
    python convert_paper_weights.py --only cube pusht # multiple envs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from jepa import JEPA  # noqa: E402
from module import ARPredictor, Embedder, MLP  # noqa: E402
import stable_pretraining as spt  # noqa: E402

BASELINE_PAPER = REPO_ROOT / "baseline_paper"
ENVS = ["pusht", "tworooms", "cube", "reacher"]


def build_model(cfg: dict) -> JEPA:
    enc_cfg = cfg["encoder"]
    encoder = spt.backbone.utils.vit_hf(
        size=enc_cfg["size"],
        patch_size=enc_cfg["patch_size"],
        image_size=enc_cfg["image_size"],
        pretrained=enc_cfg.get("pretrained", False),
        use_mask_token=enc_cfg.get("use_mask_token", False),
    )

    pred_cfg = cfg["predictor"]
    predictor = ARPredictor(
        num_frames=pred_cfg["num_frames"],
        input_dim=pred_cfg["input_dim"],
        hidden_dim=pred_cfg["hidden_dim"],
        output_dim=pred_cfg["output_dim"],
        depth=pred_cfg["depth"],
        heads=pred_cfg["heads"],
        mlp_dim=pred_cfg["mlp_dim"],
        dim_head=pred_cfg["dim_head"],
        dropout=pred_cfg["dropout"],
        emb_dropout=pred_cfg["emb_dropout"],
    )

    ae_cfg = cfg["action_encoder"]
    action_encoder = Embedder(
        input_dim=ae_cfg["input_dim"],
        emb_dim=ae_cfg["emb_dim"],
    )

    proj_cfg = cfg["projector"]
    projector = MLP(
        input_dim=proj_cfg["input_dim"],
        hidden_dim=proj_cfg["hidden_dim"],
        output_dim=proj_cfg["output_dim"],
        norm_fn=torch.nn.BatchNorm1d,
    )

    pp_cfg = cfg["pred_proj"]
    pred_proj = MLP(
        input_dim=pp_cfg["input_dim"],
        hidden_dim=pp_cfg["hidden_dim"],
        output_dim=pp_cfg["output_dim"],
        norm_fn=torch.nn.BatchNorm1d,
    )

    return JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )


def convert(env: str) -> Path:
    src_dir = BASELINE_PAPER / env
    config_path = src_dir / "config.json"
    weights_path = src_dir / "weights.pt"

    assert config_path.exists(), f"Missing {config_path}"
    assert weights_path.exists(), f"Missing {weights_path}"

    out_path = src_dir / "lewm_paper_object.ckpt"
    if out_path.exists():
        print(f"  [skip] {env}: {out_path.name} already exists", flush=True)
        return out_path

    print(f"  [convert] {env}", flush=True)

    with config_path.open() as f:
        cfg = json.load(f)

    model = build_model(cfg)

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    torch.save(model, out_path)
    size_mb = out_path.stat().st_size / 1e6
    print(f"  [saved] {out_path} ({size_mb:.1f} MB)", flush=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=ENVS,
        metavar="ENV",
        help="Envs to convert (pusht, tworooms, cube, reacher). Defaults to all.",
    )
    args = parser.parse_args()

    envs = args.only or ENVS
    for env in envs:
        try:
            convert(env)
        except Exception as exc:
            print(f"  [error] {env}: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
