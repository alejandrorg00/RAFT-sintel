# -*- coding: utf-8 -*-
"""
Visualize what the FlyVis CNN actually receives.

This script creates a GIF showing, for each temporal window:
1) original RGB video frames
2) luminance sampled on the hex lattice
3) the actual cartesian map that enters the CNN (native 31x31, zoomed for display)
4) original optical flow
5) optical flow sampled on the hex lattice
6) the cartesian flow target (native 31x31, zoomed for display)

Important:
- The CNN does NOT receive a 436x1024 "reconstruction".
- It receives the native small sparse cartesian map produced by
  RegularHexToCartesianMap, i.e. roughly [B, n_frames, 31, 31].

Run from the RAFT repo root, or from the folder containing:
    visualize_preprocessing.py
    baseline_cnn.py
    hexRenderer.py

Example:
    python visualize_preprocessing.py --scene alley_1 --dstype clean --n_frames 5 --max_windows 12

Output:
    debug_cnn_input_<scene>_<dstype>_nf<n_frames>.gif
"""

from pathlib import Path
import argparse
from glob import glob

import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.colors import hsv_to_rgb
import numpy as np
import torch
import sys
# ---------------------------------------------------------------------
# Imports from this repo
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
CORE_DIR = REPO_ROOT / "core"
sys.path.insert(0, str(CORE_DIR))

from datasets import MpiSintel, FLYVIS_TRAIN_SCENES
from flyvis_preprocessing.hexRenderer import BoxEye
from flyvis_preprocessing.baseline_cnn import RegularHexToCartesianMap


# ------------------------------------------------------------
# IO
# ------------------------------------------------------------
def read_png_rgb(path):
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    return img[..., :3].astype(np.uint8)


def read_flo(path):
    with open(path, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        if magic != 202021.25:
            raise ValueError(f"Invalid .flo file: {path}")
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
    return data.reshape(h, w, 2)


# ------------------------------------------------------------
# Color / conversion helpers
# ------------------------------------------------------------
def rgb_to_luminance_np(img_rgb):
    img = img_rgb.astype(np.float32)
    return 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]


def rgb_to_luminance_torch(img_rgb_chw):
    return (
        0.2989 * img_rgb_chw[0]
        + 0.5870 * img_rgb_chw[1]
        + 0.1140 * img_rgb_chw[2]
    )


def flow_to_rgb(flow, clip_flow=None, eps=1e-8):
    """
    flow: [H, W, 2]
    returns: [H, W, 3] in [0,1]
    """
    u = flow[..., 0]
    v = flow[..., 1]

    rad = np.sqrt(u ** 2 + v ** 2)
    ang = np.arctan2(v, u)  # [-pi, pi]

    if clip_flow is not None:
        rad = np.clip(rad, 0, clip_flow)
        rad_norm = rad / max(clip_flow, eps)
    else:
        rad_norm = rad / max(np.percentile(rad, 99), eps)

    hue = (ang + np.pi) / (2 * np.pi)
    sat = np.ones_like(hue)
    val = np.clip(rad_norm, 0, 1)

    hsv = np.stack([hue, sat, val], axis=-1)
    rgb = hsv_to_rgb(hsv)
    return rgb


def hex_flow_to_rgb(hex_uv, clip_flow=None, eps=1e-8):
    """
    hex_uv: [2, N]
    returns rgb colors [N, 3]
    """
    u = hex_uv[0]
    v = hex_uv[1]
    rad = np.sqrt(u ** 2 + v ** 2)
    ang = np.arctan2(v, u)

    if clip_flow is not None:
        rad = np.clip(rad, 0, clip_flow)
        rad_norm = rad / max(clip_flow, eps)
    else:
        rad_norm = rad / max(np.percentile(rad, 99), eps)

    hue = (ang + np.pi) / (2 * np.pi)
    sat = np.ones_like(hue)
    val = np.clip(rad_norm, 0, 1)
    hsv = np.stack([hue, sat, val], axis=-1)
    return hsv_to_rgb(hsv)


def tensor_from_hwc(img):
    return torch.from_numpy(img).permute(2, 0, 1).float()


def upsample_nearest_np(img, scale=10):
    """
    img:
      [H, W] or [H, W, 3]
    """
    if img.ndim == 2:
        return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    elif img.ndim == 3:
        return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    else:
        raise ValueError("Expected 2D or 3D image")


# ------------------------------------------------------------
# FlyVis preprocessing
# ------------------------------------------------------------
def move_boxeye_to_device(eye, device):
    eye.conv = eye.conv.to(device)
    eye.receptor_centers = eye.receptor_centers.to(device)
    eye.min_frame_size = eye.min_frame_size.to(device)
    return eye


def sample_boxeye_channels(chw, eye):
    """
    chw: [C, H, W]
    returns hexals: [1, 1, C, 721]
    """
    out = []
    for c in range(chw.shape[0]):
        seq = chw[c][None, None]  # [1,1,H,W]
        h = eye(seq, ftype="mean", hex_sample=True)  # [1,1,1,721]
        out.append(h)
    return torch.cat(out, dim=2)  # [1,1,C,721]


def hex_to_cart_native(hexals, mapper):
    """
    hexals: [1,1,C,721]
    returns:
        [C, Hc, Wc] if C>1
        [Hc, Wc] if C==1
    """
    cart = mapper(hexals)
    if cart.ndim == 4:
        # [1,1,H,W]
        return cart[0, 0]
    elif cart.ndim == 5:
        # [1,1,C,H,W]
        return cart[0, 0]
    else:
        raise RuntimeError(f"Unexpected cart shape {cart.shape}")


# ------------------------------------------------------------
# Display helpers
# ------------------------------------------------------------
def resize_rgb_strip(images_rgb, target_h=120):
    """
    images_rgb: list of [H,W,3]
    returns one horizontal strip [H2, Wsum, 3]
    """
    out = []
    for img in images_rgb:
        h, w = img.shape[:2]
        scale = target_h / h
        target_w = max(1, int(round(w * scale)))
        x = torch.from_numpy(img).permute(2, 0, 1).float()[None] / 255.0
        x = torch.nn.functional.interpolate(
            x, size=(target_h, target_w), mode="bilinear", align_corners=False
        )[0]
        out.append(x.permute(1, 2, 0).numpy())
    return np.concatenate(out, axis=1)


def resize_flow_strip(flows_rgb, target_h=120):
    out = []
    for img in flows_rgb:
        h, w = img.shape[:2]
        scale = target_h / h
        target_w = max(1, int(round(w * scale)))
        x = torch.from_numpy(img).permute(2, 0, 1).float()[None]
        x = torch.nn.functional.interpolate(
            x, size=(target_h, target_w), mode="bilinear", align_corners=False
        )[0]
        out.append(x.permute(1, 2, 0).numpy())
    return np.concatenate(out, axis=1)


def plot_hex_strip_gray(ax, hex_list, eye, title):
    """
    hex_list: list of [721]
    """
    centers = eye.receptor_centers.detach().cpu().numpy()
    y = centers[:, 0]
    x = centers[:, 1]

    xspan = x.max() - x.min() + eye.kernel_size * 5

    for i, vals in enumerate(hex_list):
        vals = vals.detach().cpu().numpy()
        xoff = i * xspan
        ax.scatter(
            x + xoff,
            -y,
            c=vals,
            cmap="gray",
            vmin=0,
            vmax=255,
            marker="h",
            s=105,
            edgecolors="none",
        )
        ax.text(
            xoff + x.mean(),
            -y.min() + 15,
            f"f{i}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_hex_strip_flow(ax, hex_flow_list, eye, title):
    """
    hex_flow_list: list of [2,721]
    """
    centers = eye.receptor_centers.detach().cpu().numpy()
    y = centers[:, 0]
    x = centers[:, 1]

    xspan = x.max() - x.min() + eye.kernel_size * 5

    for i, uv in enumerate(hex_flow_list):
        uv = uv.detach().cpu().numpy()
        colors = hex_flow_to_rgb(uv)
        xoff = i * xspan
        ax.scatter(
            x + xoff,
            -y,
            c=colors,
            marker="h",
            s=105,
            edgecolors="none",
        )
        ax.text(
            xoff + x.mean(),
            -y.min() + 15,
            f"{i}->{i+1}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")


def make_cart_lum_strip(cart_list, scale=10):
    """
    cart_list: list of [Hc,Wc]
    """
    tiles = []
    for i, cart in enumerate(cart_list):
        x = cart.detach().cpu().numpy()
        x = np.clip(x, 0, 255) / 255.0
        x = upsample_nearest_np(x, scale=scale)
        x = np.stack([x, x, x], axis=-1)
        tiles.append(x)
    return np.concatenate(tiles, axis=1)


def make_cart_flow_strip(cart_flow_list, scale=10):
    """
    cart_flow_list: list of [2,Hc,Wc]
    """
    tiles = []
    for cart in cart_flow_list:
        uv = cart.detach().cpu().numpy().transpose(1, 2, 0)  # [H,W,2]
        rgb = flow_to_rgb(uv)
        rgb = upsample_nearest_np(rgb, scale=scale)
        tiles.append(rgb)
    return np.concatenate(tiles, axis=1)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sintel_root", type=str, default="datasets/Sintel/training")
    parser.add_argument("--scene", type=str, default="alley_1")
    parser.add_argument("--dstype", type=str, default="clean", choices=["clean", "final"])
    parser.add_argument("--n_frames", type=int, default=5)
    parser.add_argument("--max_windows", type=int, default=10)
    parser.add_argument("--extent", type=int, default=15)
    parser.add_argument("--kernel_size", type=int, default=13)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if args.out is None:
        args.out = f"debug_cnn_input_{args.scene}_{args.dstype}_nf{args.n_frames}.gif"

    image_root = Path(args.sintel_root) / args.dstype / args.scene
    flow_root = Path(args.sintel_root) / "flow" / args.scene

    image_paths = sorted(glob(str(image_root / "*.png")))
    flow_paths = sorted(glob(str(flow_root / "*.flo")))

    if len(image_paths) < args.n_frames:
        raise ValueError(
            f"Scene {args.scene} only has {len(image_paths)} frames, "
            f"but n_frames={args.n_frames}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eye = move_boxeye_to_device(BoxEye(extent=args.extent, kernel_size=args.kernel_size), device)
    to_cartesian = RegularHexToCartesianMap(extent=args.extent).to(device)

    n_windows = min(args.max_windows, len(image_paths) - args.n_frames + 1)
    gif_frames = []

    print(f"Scene: {args.scene}")
    print(f"Frames available: {len(image_paths)}")
    print(f"Flow fields available: {len(flow_paths)}")
    print(f"Rendering {n_windows} windows...")
    print("Reminder:")
    print("  CNN input before to_cartesian: [B, n_frames, 1, 721]")
    print("  CNN input after to_cartesian:  [B, n_frames, 31, 31]  (approx)")
    print("  The 31x31 maps below are zoomed only for visualization.\n")

    for t0 in range(n_windows):
        # --------------------------------------------------------
        # Build temporal window
        # --------------------------------------------------------
        rgb_window = [read_png_rgb(image_paths[t]) for t in range(t0, t0 + args.n_frames)]
        lum_window = [rgb_to_luminance_np(im) for im in rgb_window]

        flow_window = [read_flo(flow_paths[t]) for t in range(t0, t0 + args.n_frames - 1)]
        flow_rgb_window = [flow_to_rgb(f) for f in flow_window]

        # --------------------------------------------------------
        # Hex-sample video and flow
        # --------------------------------------------------------
        video_hex = []
        video_cart = []

        for lum in lum_window:
            lum_t = torch.from_numpy(lum).float().to(device)  # [H,W]
            hex_lum = eye(lum_t[None, None], ftype="mean", hex_sample=True)  # [1,1,1,721]
            cart_lum = hex_to_cart_native(hex_lum, to_cartesian)  # [Hc,Wc]
            video_hex.append(hex_lum[0, 0, 0].detach().cpu())
            video_cart.append(cart_lum.detach().cpu())

        flow_hex = []
        flow_cart = []

        for flo in flow_window:
            flo_t = torch.from_numpy(flo).permute(2, 0, 1).float().to(device)  # [2,H,W]
            hex_flo = sample_boxeye_channels(flo_t, eye)  # [1,1,2,721]
            cart_flo = hex_to_cart_native(hex_flo, to_cartesian)  # [2,Hc,Wc]
            flow_hex.append(hex_flo[0, 0].detach().cpu())   # [2,721]
            flow_cart.append(cart_flo.detach().cpu())       # [2,Hc,Wc]

        # --------------------------------------------------------
        # Build strips
        # --------------------------------------------------------
        rgb_strip = resize_rgb_strip(rgb_window, target_h=125)
        flow_strip = resize_flow_strip(flow_rgb_window, target_h=125)

        cart_video_strip = make_cart_lum_strip(video_cart, scale=10)
        cart_flow_strip = make_cart_flow_strip(flow_cart, scale=10)

        # --------------------------------------------------------
        # Figure
        # --------------------------------------------------------
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        canvas = FigureCanvas(fig)

        # Top-left: original video window
        axes[0, 0].imshow(rgb_strip)
        axes[0, 0].set_title(f"Original video window  t={t0}..{t0+args.n_frames-1}")
        axes[0, 0].axis("off")

        # Top-middle: video on hex lattice
        plot_hex_strip_gray(
            axes[0, 1],
            video_hex,
            eye,
            title="Luminance sampled on hex lattice\n(before CNN cartesian embedding)"
        )

        # Top-right: actual CNN input maps
        axes[0, 2].imshow(cart_video_strip, cmap="gray", vmin=0, vmax=1)
        axes[0, 2].set_title(
            "Actual CNN input after RegularHexToCartesianMap\n"
            f"native shape ≈ [n_frames={args.n_frames}, 31, 31], zoomed for display"
        )
        axes[0, 2].axis("off")

        # Bottom-left: original flow window
        axes[1, 0].imshow(flow_strip)
        axes[1, 0].set_title(f"Original optical flow  t={t0}->{t0+args.n_frames-2}")
        axes[1, 0].axis("off")

        # Bottom-middle: flow on hex lattice
        plot_hex_strip_flow(
            axes[1, 1],
            flow_hex,
            eye,
            title="Optical flow sampled on hex lattice"
        )

        # Bottom-right: cartesian flow target
        axes[1, 2].imshow(cart_flow_strip)
        axes[1, 2].set_title(
            "Flow target in cartesian embedding\n"
            f"native shape ≈ [n_pairs={args.n_frames-1}, 2, 31, 31], zoomed"
        )
        axes[1, 2].axis("off")

        fig.suptitle(
            f"FlyVis CNN preprocessing  |  scene={args.scene}  |  dstype={args.dstype}  |  n_frames={args.n_frames}\n"
            "Top row = video / Bottom row = optic flow",
            fontsize=14
        )
        plt.tight_layout(rect=[0, 0, 1, 0.94])

        canvas.draw()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        frame = buf.reshape(h, w, 4)[..., :3].copy()
        gif_frames.append(frame)
        plt.close(fig)

        print(f"Window {t0+1}/{n_windows} done")

    imageio.mimsave(args.out, gif_frames, fps=args.fps)
    print(f"\nSaved GIF to: {args.out}")


if __name__ == "__main__":
    main()