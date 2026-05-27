# -*- coding: utf-8 -*-
"""
Visualize FlyVis-style hex preprocessing for RAFT.

Run from the RAFT-sintel repo root:

    conda activate raft_env
    python visualize_preprocessing.py

Plots:
    Original Sintel RGB | Hexals Sintel RGB | Resized hex-cartesian RGB
    Original Sintel lum | Hexals Sintel lum | Resized hex-cartesian lum
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Imports from this repo
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
CORE_DIR = REPO_ROOT / "core"
sys.path.insert(0, str(CORE_DIR))

from datasets import MpiSintel, FLYVIS_TRAIN_SCENES
from flyvis_preprocessing.hexRenderer import BoxEye
from flyvis_preprocessing.baseline_cnn import RegularHexToCartesianMap


def move_boxeye_to_device(eye, device):
    """Move all BoxEye tensors/modules used during rendering to the selected device."""
    eye.conv = eye.conv.to(device)
    eye.receptor_centers = eye.receptor_centers.to(device)
    eye.min_frame_size = eye.min_frame_size.to(device)
    return eye


def rgb_to_luminance_torch(img):
    """Convert RGB torch image [3,H,W], range 0..255, to luminance [H,W]."""
    return 0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]


def tensor_rgb_to_numpy(img):
    """Convert torch [3,H,W], 0..255 to numpy [H,W,3], 0..1."""
    img = img.detach().cpu().float().clamp(0, 255) / 255.0
    return img.permute(1, 2, 0).numpy()


def tensor_gray_to_numpy(img):
    """Convert torch [H,W] or [1,H,W] to numpy [H,W]."""
    img = img.detach().cpu().float()
    while img.ndim > 2:
        img = img.squeeze(0)
    return img.numpy()


def sample_boxeye_single_channel(channel_2d, eye):
    """Sample one [H,W] channel with BoxEye.

    Returns:
        hexals: [1, 1, 1, 721]
    """
    seq = channel_2d[None, None]
    return eye(seq, ftype="mean", hex_sample=True)


def sample_boxeye_rgb(img_rgb, eye):
    """Sample RGB image [3,H,W] channel-wise with BoxEye.

    Returns:
        hexals_rgb: [1, 1, 3, 721]
    """
    hex_channels = []
    for c in range(3):
        h = sample_boxeye_single_channel(img_rgb[c], eye)  # [1,1,1,721]
        hex_channels.append(h)

    return torch.cat(hex_channels, dim=2)  # [1,1,3,721]


def hexals_to_resized_cartesian(hexals, to_cartesian, size_hw):
    """Convert hexals to regular cartesian map and resize to original frame size.

    Args:
        hexals:
            [1,1,1,721] for luminance
            [1,1,3,721] for RGB
        to_cartesian:
            RegularHexToCartesianMap
        size_hw:
            (H, W)

    Returns:
        resized:
            [1,H,W] for luminance
            [3,H,W] for RGB
        raw_cart:
            [1,31,31] or [3,31,31]
    """
    H, W = size_hw

    cart = to_cartesian(hexals)

    # Cases:
    # luminance after RegularHexToCartesianMap can be [1,1,31,31]
    # RGB is [1,1,3,31,31]
    if cart.ndim == 4:
        # [samples, frames, Hc, Wc] -> [1, Hc, Wc]
        raw_cart = cart[0, 0][None]
    elif cart.ndim == 5:
        # [samples, frames, channels, Hc, Wc] -> [C, Hc, Wc]
        raw_cart = cart[0, 0]
    else:
        raise RuntimeError(f"Unexpected cartesian map shape: {cart.shape}")

    resized = F.interpolate(
        raw_cart[None],
        size=(H, W),
        mode="nearest",
    )[0]

    return resized, raw_cart


def plot_hexals_rgb(ax, hexals_rgb, eye, title):
    """Plot RGB hexals spatially using receptor coordinates."""
    values = hexals_rgb.detach().cpu()[0, 0]  # [3,721]
    colors = values.T.clamp(0, 255) / 255.0   # [721,3]

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
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.axis("off")


def plot_hexals_lum(ax, hexals_lum, eye, title):
    """Plot luminance hexals spatially using receptor coordinates."""
    values = hexals_lum.detach().cpu().view(-1).numpy()

    centers = eye.receptor_centers.detach().cpu()
    y = centers[:, 0].numpy()
    x = centers[:, 1].numpy()

    ax.scatter(
        x,
        -y,
        c=values,
        marker="h",
        s=115,
        cmap="gray",
        vmin=0,
        vmax=255,
        edgecolors="none",
    )
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.axis("off")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Debug visualization running on {device}.")

    dataset = MpiSintel(
        aug_params=None,
        split="training",
        dstype="clean",
        scenes=FLYVIS_TRAIN_SCENES,
        input_mode="rgb",
    )

    sample_idx = 0
    img1, img2, flow, valid = dataset[sample_idx]

    print("\nOriginal dataset tensors:")
    print("img1:", img1.shape, img1.dtype, img1.device, img1.min().item(), img1.max().item())
    print("img2:", img2.shape, img2.dtype, img2.device, img2.min().item(), img2.max().item())
    print("flow:", flow.shape, flow.dtype, flow.device)
    print("valid:", valid.shape, valid.dtype, valid.device)

    img_rgb = img1.to(device, non_blocking=True)
    H, W = img_rgb.shape[-2:]

    eye = move_boxeye_to_device(BoxEye(extent=15, kernel_size=13), device)
    to_cartesian = RegularHexToCartesianMap(extent=15).to(device)

    # -----------------------------------------------------------------
    # RGB branch
    # -----------------------------------------------------------------
    hexals_rgb = sample_boxeye_rgb(img_rgb, eye)  # [1,1,3,721]
    resized_rgb, raw_cart_rgb = hexals_to_resized_cartesian(
        hexals_rgb,
        to_cartesian,
        size_hw=(H, W),
    )

    # -----------------------------------------------------------------
    # Luminance branch
    # -----------------------------------------------------------------
    img_lum = rgb_to_luminance_torch(img_rgb)  # [H,W]
    hexals_lum = sample_boxeye_single_channel(img_lum, eye)  # [1,1,1,721]
    resized_lum, raw_cart_lum = hexals_to_resized_cartesian(
        hexals_lum,
        to_cartesian,
        size_hw=(H, W),
    )

    print("\nRGB hex output:")
    print("hexals_rgb:", hexals_rgb.shape, hexals_rgb.dtype, hexals_rgb.device)
    print("raw_cart_rgb:", raw_cart_rgb.shape, raw_cart_rgb.dtype, raw_cart_rgb.device)
    print("resized_rgb:", resized_rgb.shape, resized_rgb.dtype, resized_rgb.device)

    print("\nLuminance hex output:")
    print("hexals_lum:", hexals_lum.shape, hexals_lum.dtype, hexals_lum.device)
    print("raw_cart_lum:", raw_cart_lum.shape, raw_cart_lum.dtype, raw_cart_lum.device)
    print("resized_lum:", resized_lum.shape, resized_lum.dtype, resized_lum.device)

    # -----------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    # Row 1: RGB
    axes[0, 0].imshow(tensor_rgb_to_numpy(img_rgb))
    axes[0, 0].set_title("Original Sintel RGB")
    axes[0, 0].axis("off")

    plot_hexals_rgb(
        axes[0, 1],
        hexals_rgb,
        eye,
        title="Hexals Sintel RGB\nspatial receptor lattice",
    )

    axes[0, 2].imshow(tensor_rgb_to_numpy(resized_rgb))
    axes[0, 2].set_title(f"Resized hex-cartesian RGB\nshape={tuple(resized_rgb.shape[-2:])}")
    axes[0, 2].axis("off")

    # Row 2: luminance
    axes[1, 0].imshow(tensor_gray_to_numpy(img_lum), cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("Original Sintel luminance")
    axes[1, 0].axis("off")

    plot_hexals_lum(
        axes[1, 1],
        hexals_lum,
        eye,
        title="Hexals Sintel luminance\nspatial receptor lattice",
    )

    axes[1, 2].imshow(tensor_gray_to_numpy(resized_lum), cmap="gray", vmin=0, vmax=255)
    axes[1, 2].set_title(f"Resized hex-cartesian luminance\nshape={tuple(resized_lum.shape[-2:])}")
    axes[1, 2].axis("off")

    plt.tight_layout()

    out_dir = REPO_ROOT / "debug_hex"
    out_dir.mkdir(exist_ok=True)

    out_path = out_dir / "hex_preprocessing_rgb_vs_lum.png"
    plt.savefig(out_path, dpi=200)

    print(f"\nSaved figure to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()