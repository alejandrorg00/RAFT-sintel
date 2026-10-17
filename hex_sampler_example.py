#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a compact publication-quality figure illustrating the FlyVis hexagonal
sampler used for the RAFT matched control.

Figure layout: 2 rows x 3 columns
  Row 1: RGB Raw Sintel | LUM Hex Samples | hexRAFT Input
  Row 2: Optic Flow Raw Sintel | Optic Flow Hex Sintel Target | Optic Flow hexRAFT Sintel Target

The script saves the same figure as PNG, PDF and SVG.
Place this file in the root of the RAFT-sintel repository.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["figure.titlesize"] = 16

# -----------------------------------------------------------------------------
# Repo-local imports
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
CORE_DIR = REPO_ROOT / "core"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CORE_DIR))

from datasets import MpiSintel  # noqa: E402
from flyvis_preprocessing.raft_hex_input import RAFTFlyVisHexInput  # noqa: E402
from utils import flow_viz  # noqa: E402


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def shape_str(x: torch.Tensor) -> str:
    return "[" + ", ".join(str(v) for v in x.shape) + "]"


def tensor_rgb_to_numpy(img: torch.Tensor) -> np.ndarray:
    img = img.detach().cpu().float()
    if img.max() > 1.5:
        img = img.clamp(0, 255) / 255.0
    else:
        img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def tensor_gray_to_numpy(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().float()
    while x.ndim > 2:
        x = x.squeeze(0)
    if x.max() > 1.5:
        x = x.clamp(0, 255) / 255.0
    else:
        x = x.clamp(0, 1)
    return x.numpy()


def rgb_to_luminance_tensor(img: torch.Tensor) -> torch.Tensor:
    img = img.float()
    y = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    return y.unsqueeze(0)


def flow_tensor_to_rgb(flow: torch.Tensor) -> np.ndarray:
    flow_np = flow.detach().cpu().float().permute(1, 2, 0).numpy()
    return flow_viz.flow_to_image(flow_np).astype(np.uint8)


def flow_hex_to_rgb(hex_flow: torch.Tensor) -> np.ndarray:
    flow_values = hex_flow.detach().cpu()[0, 0]  # [2, N]
    flow_np = flow_values.T.numpy()[None]        # [1, N, 2]
    rgb = flow_viz.flow_to_image(flow_np).astype(np.uint8)[0]
    return rgb / 255.0


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def plot_image(ax, img: torch.Tensor, title: str) -> None:
    ax.imshow(tensor_rgb_to_numpy(img), interpolation="nearest")
    ax.set_title(title, fontsize=18)
    ax.axis("off")


def plot_gray(ax, img: torch.Tensor, title: str) -> None:
    ax.imshow(tensor_gray_to_numpy(img), cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(title, fontsize=18)
    ax.axis("off")


def plot_flow(ax, flow: torch.Tensor, title: str) -> None:
    ax.imshow(flow_tensor_to_rgb(flow), interpolation="nearest")
    ax.set_title(title, fontsize=18)
    ax.axis("off")


def plot_hex_gray(ax, hexals: torch.Tensor, eye, title: str) -> None:
    values = hexals.detach().cpu()[0, 0][0]      # [N]
    centers = eye.receptor_centers.detach().cpu()
    y = centers[:, 0].numpy()
    x = centers[:, 1].numpy()

    ax.scatter(
        x,
        -y,
        c=values.numpy(),
        marker="h",
        s=120,
        cmap="gray",
        vmin=0,
        vmax=255,
        edgecolors="none",
    )
    ax.set_title(title, fontsize=18)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_hex_flow(ax, hex_flow: torch.Tensor, eye, title: str) -> None:
    colors = flow_hex_to_rgb(hex_flow)
    centers = eye.receptor_centers.detach().cpu()
    y = centers[:, 0].numpy()
    x = centers[:, 1].numpy()

    ax.scatter(x, -y, c=colors, marker="h", s=120, edgecolors="none")
    ax.set_title(title, fontsize=18)
    ax.set_aspect("equal")
    ax.axis("off")


# -----------------------------------------------------------------------------
# Explicit preprocessing stages
# -----------------------------------------------------------------------------
@torch.no_grad()
def make_explicit_stages(preprocessor: RAFTFlyVisHexInput,
                         img_rgb: torch.Tensor,
                         flow: torch.Tensor,
                         valid: torch.Tensor,
                         output_size: int) -> dict[str, torch.Tensor]:
    device = preprocessor.device
    img_rgb = img_rgb.to(device)
    flow = flow.to(device)
    valid = valid.to(device).float()

    # RGB raw -> luminance
    img_lum = rgb_to_luminance_tensor(img_rgb)  # [1,H,W]

    # Intermediate hexals
    lum_hexals = preprocessor._sample_channels(img_lum)     # [1,1,1,721]
    flow_hexals = preprocessor._sample_channels(flow)       # [1,1,2,721]

    # Final RAFT-space image input for the luminance model
    lum32 = preprocessor._hexals_to_cart32(lum_hexals)      # [1,32,32]
    lum_raft_1ch = preprocessor._resize_image_like(lum32)   # [1,O,O]
    lum_raft = lum_raft_1ch.repeat(3, 1, 1)                 # [3,O,O]

    # Final RAFT-space flow target
    valid_raft = preprocessor.valid_to_raft_mask(valid)     # [O,O]
    flow_raft = preprocessor.flow_to_raft_target(flow)      # [2,O,O]
    flow_raft = flow_raft * valid_raft[None]

    return {
        "img_rgb": img_rgb.detach().cpu(),
        "img_lum": img_lum.detach().cpu(),
        "flow": flow.detach().cpu(),
        "lum_hexals": lum_hexals.detach().cpu(),
        "flow_hexals": flow_hexals.detach().cpu(),
        "lum_raft_1ch": lum_raft_1ch.detach().cpu(),
        "lum_raft": lum_raft.detach().cpu(),
        "flow_raft": flow_raft.detach().cpu(),
        "output_size": output_size,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Create the RAFT FlyVis hex-sampler figure.")
    parser.add_argument("--sintel-root", type=str, default="datasets/Sintel",
                        help="Path to the Sintel dataset root.")
    parser.add_argument("--scene", type=str, default="alley_1",
                        help="Sintel training scene to visualize.")
    parser.add_argument("--dstype", type=str, default="clean", choices=["clean", "final"],
                        help="Sintel pass.")
    parser.add_argument("--sample-idx", type=int, default=17,
                        help="Pair index within the selected scene.")
    parser.add_argument("--seed", type=int, default=12345,
                        help="Seed for deterministic loading.")
    parser.add_argument("--output-size", type=int, default=256,
                        help="Output resolution fed to RAFT.")
    parser.add_argument("--kernel-size", type=int, default=13,
                        help="BoxEye kernel size.")
    parser.add_argument("--extent", type=int, default=15,
                        help="Hexagonal extent. extent=15 gives 721 hexals.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for preprocessing.")
    parser.add_argument("--output-dir", type=str, default="figures",
                        help="Directory where figures will be saved.")
    parser.add_argument("--basename", type=str, default="hex_sampler_example",
                        help="Base filename without extension.")
    parser.add_argument("--dpi", type=int, default=600,
                        help="Raster export DPI.")
    args = parser.parse_args()

    seed_everything(args.seed)

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = MpiSintel(
        aug_params=None,
        split="training",
        root=args.sintel_root,
        dstype=args.dstype,
        scenes=[args.scene],
        input_mode="rgb",
    )

    img1, img2, flow, valid = dataset[args.sample_idx]
    del img2

    preprocessor = RAFTFlyVisHexInput(
        extent=args.extent,
        kernel_size=args.kernel_size,
        output_size=args.output_size,
        device=args.device,
    )

    stages = make_explicit_stages(preprocessor, img1, flow, valid, args.output_size)

    # -------------------------------------------------------------------------
    # Figure layout: 2 rows x 3 columns
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), dpi=args.dpi)
    fig.patch.set_facecolor("white")

    plot_image(
        axes[0, 0],
        stages["img_rgb"],
        f"Raw Sintel\n{shape_str(stages['img_rgb'])}",
    )
    plot_hex_gray(
        axes[0, 1],
        stages["lum_hexals"],
        preprocessor.eye,
        f"Hex Samples\n{shape_str(stages['lum_hexals'])}",
    )
    plot_gray(
        axes[0, 2],
        stages["lum_raft_1ch"],
        f"hexRAFT Input\n{shape_str(stages['lum_raft'])}",
    )

    plot_flow(
        axes[1, 0],
        stages["flow"],
        f"Optic Flow Raw Sintel\n{shape_str(stages['flow'])}",
    )
    plot_hex_flow(
        axes[1, 1],
        stages["flow_hexals"],
        preprocessor.eye,
        f"Optic Flow Hex Sintel \n{shape_str(stages['flow_hexals'])}",
    )
    plot_flow(
        axes[1, 2],
        stages["flow_raft"],
        f"Optic Flow hexRAFT Sintel \n{shape_str(stages['flow_raft'])}",
    )

    # fig.suptitle(
    #     f"RAFT FlyVis-hex preprocessing example | scene={args.scene}, pair={args.sample_idx + 1}, dstype={args.dstype}",
    #     fontsize=20,
    #     y=0.985,
    # )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.align_titles(axes)

    base = output_dir / args.basename
    png_path = str(base.with_suffix(".png"))
    pdf_path = str(base.with_suffix(".pdf"))
    svg_path = str(base.with_suffix(".svg"))

    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Saved:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")
    print(f"  {svg_path}")


if __name__ == "__main__":
    main()