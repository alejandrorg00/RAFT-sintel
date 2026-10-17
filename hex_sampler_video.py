#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a compact publication-quality figure illustrating the FlyVis hexagonal
sampler used for the RAFT matched control.

Figure layout: 2 rows x 3 columns
  Row 1: RGB Raw Sintel | LUM Hex Samples | hexRAFT Input
  Row 2: Optic Flow Raw Sintel | Optic Flow Hex Sintel Target | Optic Flow hexRAFT Sintel Target

The script animates an entire Sintel scene and saves GIF and MP4.
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
import imageio.v2 as imageio

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
    parser = argparse.ArgumentParser(
        description="Animate a complete Sintel scene through the FlyVis hex sampler."
    )
    parser.add_argument("--sintel-root", default="datasets/Sintel")
    parser.add_argument("--scene", default="alley_1")
    parser.add_argument("--dstype", choices=("clean", "final"), default="clean")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--basename", default=None,
                        help="Default: hex_sampler_<scene>_<dstype>")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=120,
                        help="Video rasterization DPI; 120 keeps files practical.")
    parser.add_argument("--output-size", type=int, default=256)
    parser.add_argument("--kernel-size", type=int, default=13)
    parser.add_argument("--extent", type=int, default=15)
    parser.add_argument("--start", type=int, default=0,
                        help="First frame pair index (zero-based).")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Optional limit for testing (default: whole scene).")
    args = parser.parse_args()
    if args.fps <= 0 or args.dpi <= 0:
        parser.error("--fps and --dpi must be positive")
    seed_everything(12345)

    dataset = MpiSintel(
        aug_params=None,
        split="training",
        root=str((REPO_ROOT / args.sintel_root).resolve()) if not Path(args.sintel_root).is_absolute() else args.sintel_root,
        dstype=args.dstype,
        scenes=[args.scene],
        input_mode="rgb",
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No frame pairs found for scene {args.scene!r} in {args.sintel_root}")
    if args.start < 0 or args.start >= len(dataset):
        parser.error(f"--start must be in [0, {len(dataset)-1}]")

    stop = len(dataset)
    if args.max_frames is not None:
        if args.max_frames <= 0:
            parser.error("--max-frames must be positive")
        stop = min(stop, args.start + args.max_frames)

    preprocessor = RAFTFlyVisHexInput(
        extent=args.extent,
        kernel_size=args.kernel_size,
        output_size=args.output_size,
        device=args.device,
    )

    out_dir = (REPO_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base = args.basename or f"hex_sampler_{args.scene}_{args.dstype}"
    gif_path = out_dir / f"{base}.gif"
    mp4_path = out_dir / f"{base}.mp4"

    # Create axes and titles only once, so the entire video has a stable layout.
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), dpi=args.dpi)
    fig.patch.set_facecolor("white")
    headings = [
        "Raw Sintel", "Hex Samples", "hexRAFT Input",
        "Optic Flow Raw Sintel", "Optic Flow Hex Sintel Target",
        "Optic Flow hexRAFT Sintel Target",
    ]

    def as_frame(idx: int):
        img1, _, flow, valid = dataset[idx]
        stages = make_explicit_stages(preprocessor, img1, flow, valid, args.output_size)
        return stages

    first = as_frame(args.start)
    labels = [
        f"{headings[0]}\n{shape_str(first['img_rgb'])}",
        f"{headings[1]}\n{shape_str(first['lum_hexals'])}",
        f"{headings[2]}\n{shape_str(first['lum_raft'])}",
        f"{headings[3]}\n{shape_str(first['flow'])}",
        f"{headings[4]}\n{shape_str(first['flow_hexals'])}",
        f"{headings[5]}\n{shape_str(first['flow_raft'])}",
    ]
    ax = axes.ravel()
    image_artists = [
        ax[0].imshow(tensor_rgb_to_numpy(first["img_rgb"]), interpolation="nearest"),
        None,
        ax[2].imshow(tensor_gray_to_numpy(first["lum_raft_1ch"]),
                     cmap="gray", vmin=0, vmax=1, interpolation="nearest"),
        ax[3].imshow(flow_tensor_to_rgb(first["flow"]), interpolation="nearest"),
        None,
        ax[5].imshow(flow_tensor_to_rgb(first["flow_raft"]), interpolation="nearest"),
    ]
    centers = preprocessor.eye.receptor_centers.detach().cpu()
    y = -centers[:, 0].numpy()
    x = centers[:, 1].numpy()
    hex_gray_artist = ax[1].scatter(
        x, y, c=first["lum_hexals"][0, 0, 0].numpy(), marker="h", s=120,
        cmap="gray", vmin=0, vmax=255, edgecolors="none",
    )
    hex_flow_artist = ax[4].scatter(
        x, y, c=flow_hex_to_rgb(first["flow_hexals"]), marker="h",
        s=120, edgecolors="none",
    )
    ax[1].set_aspect("equal")
    ax[4].set_aspect("equal")
    for a, title in zip(ax, labels):
        a.set_title(title, fontsize=18)
        a.axis("off")
    # Match the styling from the still figure; retain an identical axes geometry.
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.align_titles(axes)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    # H.264/yuv420p requires even dimensions. Pad at most one pixel on
    # the right/bottom, without changing the figure layout or typography.
    pad_right = width % 2
    pad_bottom = height % 2
    if pad_right or pad_bottom:
        print(f"MP4 frame: {width}x{height} -> "
              f"{width + pad_right}x{height + pad_bottom} (white padding)")

    # imageio-ffmpeg provides the executable; GIF frames are streamed incrementally.
    # GIF is large for full-resolution long clips; use MP4 for convenient playback.
    try:
        with imageio.get_writer(str(mp4_path), fps=args.fps, codec="libx264",
                                quality=8, macro_block_size=2,
                                pixelformat="yuv420p") as video, \
             imageio.get_writer(str(gif_path), mode="I", duration=1 / args.fps,
                                loop=0, subrectangles=False) as gif:
            for idx in range(args.start, stop):
                stages = first if idx == args.start else as_frame(idx)
                image_artists[0].set_data(tensor_rgb_to_numpy(stages["img_rgb"]))
                hex_gray_artist.set_array(stages["lum_hexals"][0, 0, 0].numpy())
                image_artists[2].set_data(tensor_gray_to_numpy(stages["lum_raft_1ch"]))
                image_artists[3].set_data(flow_tensor_to_rgb(stages["flow"]))
                hex_flow_artist.set_facecolors(flow_hex_to_rgb(stages["flow_hexals"]))
                image_artists[5].set_data(flow_tensor_to_rgb(stages["flow_raft"]))
                fig.canvas.draw()
                frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
                if pad_right or pad_bottom:
                    mp4_frame = np.pad(
                        frame,
                        ((0, pad_bottom), (0, pad_right), (0, 0)),
                        mode="constant", constant_values=255,
                    )
                else:
                    mp4_frame = frame
                video.append_data(mp4_frame)
                gif.append_data(frame)
                print(f"\rEncoding frame {idx - args.start + 1}/{stop - args.start}", end="", flush=True)
        print()
    finally:
        plt.close(fig)

    print(f"Saved MP4: {mp4_path}")
    print(f"Saved GIF: {gif_path}")


if __name__ == "__main__":
    main()
