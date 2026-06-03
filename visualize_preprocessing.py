# -*- coding: utf-8 -*-
"""
Visualize current RAFT FlyVis-hex preprocessing pipeline.

Current pipeline:

    MpiSintel normal RGB/LUM crop
    -> RAFTFlyVisHexInput
    -> BoxEye hexals
    -> RegularHexToCartesianMap
    -> 31x31 sparse cartesian map
    -> pad to 32x32
    -> upsample to 256x256 RAFT input / target

This script does NOT use MpiSintel(..., flyvis_hex=True).
That old path is obsolete.

Run from RAFT-sintel root:

    python visualize_gpuhex_preprocessing.py --scene alley_1 --dstype clean --sample_idx 0

With training augmentation/crop:

    python visualize_gpuhex_preprocessing.py --scene alley_1 --dstype clean --sample_idx 0 --use_train_aug
"""

from pathlib import Path
import sys
import argparse
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

torch.set_default_device("cpu")


# ---------------------------------------------------------------------
# Repo imports
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
CORE_DIR = REPO_ROOT / "core"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CORE_DIR))

from datasets import MpiSintel
from flyvis_preprocessing.raft_hex_input import RAFTFlyVisHexInput
from utils import flow_viz


# ---------------------------------------------------------------------
# Deterministic dataset access
# ---------------------------------------------------------------------
def get_item_deterministic(dataset, idx, seed=12345):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return dataset[idx]


# ---------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------
def tensor_rgb_to_numpy(img):
    """[3,H,W] torch in 0..255 or 0..1 -> [H,W,3] numpy in 0..1."""
    img = img.detach().cpu().float()

    if img.max() > 1.5:
        img = img.clamp(0, 255) / 255.0
    else:
        img = img.clamp(0, 1)

    return img.permute(1, 2, 0).numpy()


def tensor_gray_to_numpy(x):
    x = x.detach().cpu().float()
    while x.ndim > 2:
        x = x.squeeze(0)
    return x.numpy()


def flow_tensor_to_rgb(flow):
    """[2,H,W] torch -> [H,W,3] uint8."""
    flow_np = flow.detach().cpu().float().permute(1, 2, 0).numpy()
    return flow_viz.flow_to_image(flow_np).astype(np.uint8)


def flow_hex_to_rgb(hex_flow):
    """hex_flow [1,1,2,N] -> RGB colors [N,3]."""
    flow_values = hex_flow.detach().cpu()[0, 0]  # [2,N]
    flow_np = flow_values.T.numpy()[None]        # [1,N,2]
    rgb = flow_viz.flow_to_image(flow_np).astype(np.uint8)[0]
    return rgb / 255.0


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def show_rgb(ax, img, title):
    ax.imshow(tensor_rgb_to_numpy(img), interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")


def show_flow(ax, flow, title):
    ax.imshow(flow_tensor_to_rgb(flow), interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")


def show_mask(ax, mask, title):
    ax.imshow(tensor_gray_to_numpy(mask), cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")


def plot_hex_rgb(ax, hexals, eye, title):
    """hexals [1,1,3,N]."""
    values = hexals.detach().cpu()[0, 0]          # [3,N]
    colors = values.T.clamp(0, 255) / 255.0       # [N,3]

    centers = eye.receptor_centers.detach().cpu()
    y = centers[:, 0].numpy()
    x = centers[:, 1].numpy()

    ax.scatter(
        x,
        -y,
        c=colors.numpy(),
        marker="h",
        s=115,
        edgecolors="none",
    )
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_hex_lum(ax, hexals, eye, title):
    """hexals [1,1,1,N] or [1,1,3,N]."""
    values = hexals.detach().cpu()[0, 0]  # [C,N]
    values = values[0]                    # [N]

    centers = eye.receptor_centers.detach().cpu()
    y = centers[:, 0].numpy()
    x = centers[:, 1].numpy()

    ax.scatter(
        x,
        -y,
        c=values.numpy(),
        marker="h",
        s=115,
        cmap="gray",
        vmin=0,
        vmax=255,
        edgecolors="none",
    )
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_hex_flow(ax, hex_flow, eye, title):
    """hex_flow [1,1,2,N]."""
    colors = flow_hex_to_rgb(hex_flow)

    centers = eye.receptor_centers.detach().cpu()
    y = centers[:, 0].numpy()
    x = centers[:, 1].numpy()

    ax.scatter(
        x,
        -y,
        c=colors,
        marker="h",
        s=115,
        edgecolors="none",
    )
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")


# ---------------------------------------------------------------------
# Low-level explicit preprocessing
# ---------------------------------------------------------------------
@torch.no_grad()
def explicit_hex_pipeline(preprocessor, image, flow, valid, input_mode):
    """
    Return every stage:
        image_hexals
        flow_hexals
        valid_hexals
        image_cart32
        flow_cart32
        valid_cart32
        image_raft
        flow_raft
        valid_raft
    """

    device = preprocessor.device

    image = image.to(device)
    flow = flow.to(device)
    valid = valid.to(device).float()

    # ------------------------------------------------------------
    # 1. Intermediate hexals
    # ------------------------------------------------------------
    if input_mode == "lum":
        # Dataset has already repeated luminance over 3 channels.
        # Use only channel 0, same logic as batch_image_to_raft_input(..., lum).
        image_for_hex = image[:1]
    elif input_mode == "rgb":
        image_for_hex = image
    else:
        raise ValueError(f"Unknown input_mode: {input_mode}")

    image_hexals = preprocessor._sample_channels(image_for_hex)       # [1,1,C,N]
    flow_hexals = preprocessor._sample_channels(flow)                 # [1,1,2,N]
    valid_hexals = preprocessor.eye(valid[None, None], ftype="mean", hex_sample=True)

    # ------------------------------------------------------------
    # 2. Hex-to-cartesian 32x32 sparse maps
    #    This is where the black empty lateral / padded locations appear.
    # ------------------------------------------------------------
    image_cart32 = preprocessor._hexals_to_cart32(image_hexals)        # [C,32,32]
    flow_cart32 = preprocessor._hexals_to_cart32(flow_hexals)          # [2,32,32]
    valid_cart32 = preprocessor._hexals_to_cart32(valid_hexals)[0]     # [32,32]
    valid_cart32 = (valid_cart32 > 0.0).float()

    if input_mode == "lum":
        image_cart32_for_display = image_cart32.repeat(3, 1, 1)
    else:
        image_cart32_for_display = image_cart32

    flow_cart32_masked = flow_cart32 * valid_cart32[None]

    # ------------------------------------------------------------
    # 3. Final RAFT input / target 256x256
    # ------------------------------------------------------------
    image_raft = preprocessor.image_to_raft_input(image)
    flow_raft = preprocessor.flow_to_raft_target(flow)
    valid_raft = preprocessor.valid_to_raft_mask(valid)

    flow_raft_masked = flow_raft * valid_raft[None]

    return {
        "image_hexals": image_hexals,
        "flow_hexals": flow_hexals,
        "valid_hexals": valid_hexals,
        "image_cart32": image_cart32_for_display,
        "flow_cart32": flow_cart32_masked,
        "valid_cart32": valid_cart32,
        "image_raft": image_raft,
        "flow_raft": flow_raft_masked,
        "valid_raft": valid_raft,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scene", type=str, default="alley_1")
    parser.add_argument("--dstype", type=str, default="clean", choices=["clean", "final"])
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--use_train_aug", action="store_true")
    parser.add_argument("--input_mode", type=str, default="lum", choices=["rgb", "lum"])
    parser.add_argument("--output_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="debug_hex/gpuhex_pipeline_check.png")

    args = parser.parse_args()

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.use_train_aug:
        aug_params = {
            "crop_size": [368, 768],
            "min_scale": -0.2,
            "max_scale": 0.6,
            "do_flip": True,
        }
    else:
        aug_params = None

    dataset = MpiSintel(
        aug_params=aug_params,
        split="training",
        dstype=args.dstype,
        scenes=[args.scene],
        input_mode=args.input_mode,
    )

    img1, img2, flow, valid = get_item_deterministic(
        dataset,
        args.sample_idx,
        seed=12345 + args.sample_idx,
    )

    preprocessor = RAFTFlyVisHexInput(
        extent=15,
        kernel_size=13,
        output_size=args.output_size,
        device=args.device,
    )

    stages = explicit_hex_pipeline(
        preprocessor=preprocessor,
        image=img1,
        flow=flow,
        valid=valid,
        input_mode=args.input_mode,
    )

    # ------------------------------------------------------------
    # Print sanity shapes and values
    # ------------------------------------------------------------
    print("\nRaw dataset output")
    print("img1 :", tuple(img1.shape), img1.dtype, float(img1.min()), float(img1.max()))
    print("flow :", tuple(flow.shape), flow.dtype, float(flow.min()), float(flow.max()))
    print("valid:", tuple(valid.shape), valid.dtype, float(valid.min()), float(valid.max()))

    print("\nIntermediate/final pipeline")
    print("image_hexals:", tuple(stages["image_hexals"].shape))
    print("flow_hexals :", tuple(stages["flow_hexals"].shape))
    print("image_cart32:", tuple(stages["image_cart32"].shape))
    print("flow_cart32 :", tuple(stages["flow_cart32"].shape))
    print("valid_cart32:", tuple(stages["valid_cart32"].shape))
    print("image_raft  :", tuple(stages["image_raft"].shape))
    print("flow_raft   :", tuple(stages["flow_raft"].shape))
    print("valid_raft  :", tuple(stages["valid_raft"].shape))

    print("\nValid pixels")
    print("valid_cart32:", int(stages["valid_cart32"].sum().item()), "/", stages["valid_cart32"].numel())
    print("valid_raft  :", int(stages["valid_raft"].sum().item()), "/", stages["valid_raft"].numel())

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), dpi=120)

    title_suffix = (
        f"scene={args.scene}, pair={args.sample_idx}, dstype={args.dstype}, "
        f"input_mode={args.input_mode}, aug={args.use_train_aug}"
    )

    # Row 1: image
    show_rgb(
        axes[0, 0],
        img1,
        "Image original cartesian\nMpiSintel output",
    )

    if args.input_mode == "rgb":
        plot_hex_rgb(
            axes[0, 1],
            stages["image_hexals"],
            preprocessor.eye,
            "Image intermediate hexals\nBoxEye samples",
        )
    else:
        plot_hex_lum(
            axes[0, 1],
            stages["image_hexals"],
            preprocessor.eye,
            "Luminance intermediate hexals\nBoxEye samples",
        )

    show_rgb(
        axes[0, 2],
        stages["image_cart32"],
        "Image hex → cartesian 32×32\nblack = empty/padded positions",
    )

    show_rgb(
        axes[0, 3],
        stages["image_raft"],
        f"Image final RAFT input\n{args.output_size}×{args.output_size}",
    )

    # Row 2: flow
    show_flow(
        axes[1, 0],
        flow,
        "Flow original cartesian\nSintel target",
    )

    plot_hex_flow(
        axes[1, 1],
        stages["flow_hexals"],
        preprocessor.eye,
        "Flow intermediate hexals\nBoxEye samples",
    )

    show_flow(
        axes[1, 2],
        stages["flow_cart32"],
        "Flow hex → cartesian 32×32\nmasked by valid_cart32",
    )

    show_flow(
        axes[1, 3],
        stages["flow_raft"],
        f"Flow final RAFT target\n{args.output_size}×{args.output_size}",
    )

    # Row 3: valid mask
    show_mask(
        axes[2, 0],
        valid,
        "Valid original cartesian",
    )

    # For valid hexals, plot as luminance hexals.
    plot_hex_lum(
        axes[2, 1],
        stages["valid_hexals"],
        preprocessor.eye,
        "Valid intermediate hexals",
    )

    show_mask(
        axes[2, 2],
        stages["valid_cart32"],
        "Valid hex → cartesian 32×32\nblack = invalid/empty/padded",
    )

    show_mask(
        axes[2, 3],
        stages["valid_raft"],
        f"Valid final RAFT mask\n{args.output_size}×{args.output_size}",
    )

    fig.suptitle(
        "Current GPU-hex preprocessing check\n"
        + title_suffix
        + "\nColumns: original cartesian / hexals / hex-to-cartesian 32×32 / final RAFT 256×256",
        fontsize=12,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path)
    print("\nSaved plot to:")
    print(out_path)

    plt.show()


if __name__ == "__main__":
    main()