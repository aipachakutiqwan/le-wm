import argparse
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import stable_worldmodel as swm


def to_hwc(frame: torch.Tensor) -> np.ndarray:
    """(3, H, W) uint8 tensor -> (H, W, 3) uint8 numpy array for imshow."""
    return frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tworoom",
                    help="dataset name under STABLEWM_HOME (e.g. tworoom, pusht_expert_train)")
    ap.add_argument("--n", type=int, default=16, help="number of random frames (grid mode)")
    ap.add_argument("--num-steps", type=int, default=1,
                    help="window length (frames per sample)")
    ap.add_argument("--frameskip", type=int, default=1, help="stride between frames in a window")
    ap.add_argument("--out", default=None, help="output PNG path (default: ./<dataset>.png)")
    args = ap.parse_args()

    os.environ.setdefault("STABLEWM_HOME", os.environ.get("STABLEWM_HOME", "/stablewm-home"))
    rng = np.random.default_rng(0)

    ds = swm.data.HDF5Dataset(
        name=args.dataset, num_steps=args.num_steps, frameskip=args.frameskip,
        keys_to_load=["pixels"], keys_to_cache=[], transform=None,
    )
    print(f"dataset={args.dataset}  len={len(ds)}  num_steps={args.num_steps}  frameskip={args.frameskip}")
    title_map = {
        "cube_single_expert": "OGBench-Cube",
        "pusht_expert_train": "Push-T",
        "reacher": "Reacher",
        "tworoom": "Two Room"
    }
    

    # n random windows, show the first frame of each
    idxs = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)
    imgs = [to_hwc(ds[int(i)]["pixels"][0]) for i in idxs]
    titles = [f"#{int(i)}" for i in idxs]
    suptitle = f"{title_map[args.dataset]}"

    cols = int(np.ceil(np.sqrt(len(imgs))))
    rows = int(np.ceil(len(imgs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    axes = np.atleast_1d(axes).ravel()
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img)
        # ax.set_title(title, fontsize=8)
        ax.axis("off")
    for ax in axes[len(imgs):]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()

    out = Path(args.out) if args.out else Path(f"{args.dataset}.png")
    fig.savefig(out, dpi=700, bbox_inches="tight")
    print(f"saved {len(imgs)} frames to {out.resolve()}")


if __name__ == "__main__":
    main()
