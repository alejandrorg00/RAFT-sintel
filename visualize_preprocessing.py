# -*- coding: utf-8 -*-
"""
Visualize the real RAFT input returned by the dataloader.

Layout:
    rgb sintel        / rgb hex sintel        / rgb raft input from dataset
    lum sintel        / lum hex sintel        / lum raft input from dataset
    optic flow sintel / optic flow hex sintel / optic flow raft target from dataset

Important:
    The third column comes from:

        MpiSintel(..., flyvis_hex=True)

    Therefore it is the actual tensor returned by the dataloader and the
    tensor that train.py will feed to RAFT.

Run from the RAFT-sintel repo root:

    python visualize_preprocessing.py --scene alley_1 --dstype clean --save_mp4

To visualize training-like random crops:

    python visualize_preprocessing.py --scene alley_1 --dstype clean --use_train_aug --save_mp4

Outputs:
    debug_hex/raft_hex_dataset_3x3.gif
    debug_hex/raft_hex_dataset_3x3.mp4
"""

from pathlib import Path
import sys
import argparse
import random

import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import torch

torch.set_default_device("cpu")


# ---------------------------------------------------------------------
# Imports from this repo
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
CORE_DIR = REPO_ROOT / "core"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CORE_DIR))

from datasets import MpiSintel
from utils import flow_viz


# ---------------------------------------------------------------------
# Deterministic dataset access
# ---------------------------------------------------------------------
def get_item_deterministic(dataset, idx, seed):
    """Get dataset[idx] with deterministic random augmentation.

    This is useful when comparing:
        raw dataset output
        flyvis_hex=True dataset output

    If both are called with the same seed, they should receive the same
    random crop/augmentation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return dataset[idx]


# ---------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------
def tensor_rgb_to_numpy(img):
    """Convert torch [3,H,W] to numpy [H,W,3] in [0,1]."""
    img = img.detach().cpu().float()

    if img.max() > 1.5:
        img = img.clamp(0, 255) / 255.0
    else:
        img = img.clamp(0, 1)

    return img.permute(1, 2, 0).numpy()


def tensor_gray_to_numpy(img):
    """Convert torch [H,W] or [1,H,W] to numpy [H,W]."""
    img = img.detach().cpu().float()

    while img.ndim > 2:
        img = img.squeeze(0)

    return img.numpy()


def flow_tensor_to_rgb(flow):
    """Convert torch flow [2,H,W] to RGB using RAFT's flow_viz."""
    flow_np = flow.detach().cpu().float().permute(1, 2, 0).numpy()
    flow_img = flow_viz.flow_to_image(flow_np)
    return flow_img.astype(np.uint8)


def flow_hex_to_rgb(hex_flow):
    """Convert hex flow [2,N] to RGB colors [N,3] using RAFT's flow_viz."""
    flow_np = hex_flow.detach().cpu().float().T.numpy()  # [N, 2]
    flow_np = flow_np[None]  # [1, N, 2]

    rgb = flow_viz.flow_to_image(flow_np).astype(np.uint8)[0]  # [N, 3]
    return rgb / 255.0


def show_rgb(ax, img, title):
    ax.imshow(tensor_rgb_to_numpy(img), interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")


def show_gray(ax, img, title, vmin=None, vmax=None):
    ax.imshow(
        tensor_gray_to_numpy(img),
        cmap="gray",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.axis("off")


def show_flow(ax, flow, title):
    ax.imshow(flow_tensor_to_rgb(flow), interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")


def plot_hex_rgb(ax, hexals, eye, title):
    """Plot RGB hexals.

    Args:
        hexals: [1, 1, 3, 721]
    """
    values = hexals.detach().cpu()[0, 0]  # [3, 721]
    colors = values.T.clamp(0, 255) / 255.0  # [721, 3]

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
    """Plot luminance hexals.

    Args:
        hexals: [1, 1, 3, 721] or [1, 1, 1, 721]

    For input_mode='lum', datasets.py has already repeated luminance
    over three channels. We plot channel 0.
    """
    values = hexals.detach().cpu()[0, 0]  # [C, 721]
    values = values[0]  # [721]

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
    """Plot optical-flow hexals.

    Args:
        hex_flow: [1, 1, 2, 721]
    """
    flow_values = hex_flow.detach().cpu()[0, 0]  # [2, 721]
    colors = flow_hex_to_rgb(flow_values)

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


def make_video_frame(fig):
    """Render matplotlib figure to RGB numpy array."""
    canvas = FigureCanvas(fig)
    canvas.draw()

    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()

    return buf.reshape(h, w, 4)[..., :3].copy()


def assert_shapes(name, img1, img2, flow, valid, hex_output_size):
    """Print and assert the expected RAFT-hex shapes."""
    print(f"\n{name}")
    print("img1 :", tuple(img1.shape), img1.dtype)
    print("img2 :", tuple(img2.shape), img2.dtype)
    print("flow :", tuple(flow.shape), flow.dtype)
    print("valid:", tuple(valid.shape), valid.dtype)

    assert tuple(img1.shape) == (3, hex_output_size, hex_output_size), (
        f"{name} img1 wrong shape: {img1.shape}"
    )
    assert tuple(img2.shape) == (3, hex_output_size, hex_output_size), (
        f"{name} img2 wrong shape: {img2.shape}"
    )
    assert tuple(flow.shape) == (2, hex_output_size, hex_output_size), (
        f"{name} flow wrong shape: {flow.shape}"
    )
    assert tuple(valid.shape) == (hex_output_size, hex_output_size), (
        f"{name} valid wrong shape: {valid.shape}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scene",
        type=str,
        default="alley_1",
        help="Sintel scene to visualize, e.g. alley_1.",
    )
    parser.add_argument(
        "--dstype",
        type=str,
        default="clean",
        choices=["clean", "final"],
        help="Sintel render type.",
    )
    parser.add_argument(
        "--sample_start",
        type=int,
        default=0,
        help="First pair index inside the selected scene.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Number of pairs to render. If omitted, render the whole scene.",
    )
    parser.add_argument(
        "--use_train_aug",
        action="store_true",
        help="Use the same crop/augmentation settings as training: crop_size=[368,768].",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="GIF/MP4 frames per second.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="debug_hex/raft_hex_dataset_3x3.gif",
        help="Output GIF path relative to repo root.",
    )
    parser.add_argument(
        "--save_mp4",
        action="store_true",
        help="Also save an MP4 video.",
    )
    parser.add_argument(
        "--mp4_out",
        type=str,
        default="debug_hex/raft_hex_dataset_3x3.mp4",
        help="Output MP4 path relative to repo root.",
    )

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

    # -----------------------------------------------------------------
    # Raw Sintel datasets.
    # These are for column 1 and also for manual hex visualization.
    # -----------------------------------------------------------------
    dataset_rgb_raw = MpiSintel(
        aug_params=aug_params,
        split="training",
        dstype=args.dstype,
        scenes=[args.scene],
        input_mode="rgb",
        flyvis_hex=False,
    )

    dataset_lum_raw = MpiSintel(
        aug_params=aug_params,
        split="training",
        dstype=args.dstype,
        scenes=[args.scene],
        input_mode="lum",
        flyvis_hex=False,
    )

    # -----------------------------------------------------------------
    # Real RAFT-hex datasets.
    # These produce the actual tensors that train.py/evaluate.py will
    # feed to RAFT when flyvis_hex=True.
    # -----------------------------------------------------------------
    dataset_rgb_hex = MpiSintel(
        aug_params=aug_params,
        split="training",
        dstype=args.dstype,
        scenes=[args.scene],
        input_mode="rgb",
        flyvis_hex=True,
    )

    dataset_lum_hex = MpiSintel(
        aug_params=aug_params,
        split="training",
        dstype=args.dstype,
        scenes=[args.scene],
        input_mode="lum",
        flyvis_hex=True,
    )

    n_total = min(
        len(dataset_rgb_raw),
        len(dataset_lum_raw),
        len(dataset_rgb_hex),
        len(dataset_lum_hex),
    )

    if args.sample_start >= n_total:
        raise ValueError(
            f"sample_start={args.sample_start} is outside dataset length {n_total} "
            f"for scene={args.scene}."
        )

    if args.max_samples is None:
        sample_end = n_total
    else:
        sample_end = min(args.sample_start + args.max_samples, n_total)

    print(f"\nScene: {args.scene}")
    print(f"dstype: {args.dstype}")
    print(f"Dataset length for selected scene: {n_total}")
    print(f"Visualizing pairs: {args.sample_start} to {sample_end - 1}")
    print(f"Output GIF: {out_path}")

    if args.use_train_aug:
        print("Augmentation: training-like random crop/scale/flip")
        print(f"aug_params: {aug_params}")
    else:
        print("Augmentation: None. Visualizing full Sintel frames.")

    # Use the preprocessor attached to the actual hex dataset.
    # This guarantees that the middle-column hex visualisation uses the
    # same BoxEye/RegularHexToCartesianMap object as the dataloader.
    hex_preprocessor = dataset_rgb_hex.hex_preprocessor
    eye = hex_preprocessor.eye

    hex_output_size = hex_preprocessor.output_size
    raft_img_shape = f"[3,{hex_output_size},{hex_output_size}]"
    raft_flow_shape = f"[2,{hex_output_size},{hex_output_size}]"

    print(f"Internal RAFT-hex output size: {hex_output_size}x{hex_output_size}")

    video_frames = []

    for sample_idx in range(args.sample_start, sample_end):
        seed = 12345 + sample_idx

        # ------------------------------------------------------------
        # Raw samples.
        # If --use_train_aug is active, deterministic seeding ensures
        # raw and hex datasets use the same random crop.
        # ------------------------------------------------------------
        img1_rgb_raw, img2_rgb_raw, flow_raw, valid_raw = get_item_deterministic(
            dataset_rgb_raw,
            sample_idx,
            seed,
        )

        img1_lum_raw, img2_lum_raw, flow_lum_raw, valid_lum_raw = get_item_deterministic(
            dataset_lum_raw,
            sample_idx,
            seed,
        )

        # ------------------------------------------------------------
        # Actual RAFT-hex dataloader outputs.
        # These are the tensors that would enter RAFT.
        # ------------------------------------------------------------
        img1_rgb_raft, img2_rgb_raft, flow_rgb_raft, valid_rgb_raft = get_item_deterministic(
            dataset_rgb_hex,
            sample_idx,
            seed,
        )

        img1_lum_raft, img2_lum_raft, flow_lum_raft, valid_lum_raft = get_item_deterministic(
            dataset_lum_hex,
            sample_idx,
            seed,
        )

        if sample_idx == args.sample_start:
            print("\nRaw tensors")
            print("img1_rgb_raw:", tuple(img1_rgb_raw.shape), img1_rgb_raw.dtype)
            print("img1_lum_raw:", tuple(img1_lum_raw.shape), img1_lum_raw.dtype)
            print("flow_raw    :", tuple(flow_raw.shape), flow_raw.dtype)
            print("valid_raw   :", tuple(valid_raw.shape), valid_raw.dtype)

            assert_shapes(
                "RGB RAFT-hex dataset output",
                img1_rgb_raft,
                img2_rgb_raft,
                flow_rgb_raft,
                valid_rgb_raft,
                hex_output_size,
            )

            assert_shapes(
                "LUM RAFT-hex dataset output",
                img1_lum_raft,
                img2_lum_raft,
                flow_lum_raft,
                valid_lum_raft,
                hex_output_size,
            )

            # Extra sanity check:
            # manual call through the same preprocessor should match
            # the flyvis_hex=True dataloader output because the raw
            # and hex datasets were accessed with the same seed.
            manual_img1_rgb = hex_preprocessor.image_to_raft_input(img1_rgb_raw)
            manual_flow_rgb = hex_preprocessor.flow_to_raft_target(flow_raw)
            manual_valid_rgb = hex_preprocessor.valid_to_raft_mask(valid_raw)

            print("\nSanity check against manual preprocessing")
            print(
                "max |manual_img1_rgb - dataset_img1_rgb|:",
                (manual_img1_rgb - img1_rgb_raft).abs().max().item(),
            )
            print(
                "max |manual_flow_rgb - dataset_flow_rgb|:",
                (manual_flow_rgb - flow_rgb_raft).abs().max().item(),
            )
            print(
                "max |manual_valid_rgb - dataset_valid_rgb|:",
                (manual_valid_rgb - valid_rgb_raft).abs().max().item(),
            )

        # ------------------------------------------------------------
        # Hex visualisation for middle column.
        # This is not what enters RAFT. It is the intermediate 721-hexal
        # representation, shown for interpretability.
        # ------------------------------------------------------------
        rgb_hex = hex_preprocessor._sample_channels(img1_rgb_raw)  # [1,1,3,721]
        lum_hex = hex_preprocessor._sample_channels(img1_lum_raw)  # [1,1,3,721]
        flow_hex = hex_preprocessor._sample_channels(flow_raw)     # [1,1,2,721]

        # Mask the displayed flow target so empty positions are not shown as real data.
        flow_raft_masked = flow_rgb_raft * valid_rgb_raft[None]

        # ------------------------------------------------------------
        # 3 x 3 plot
        # ------------------------------------------------------------
        fig, axes = plt.subplots(3, 3, figsize=(12.8, 12), dpi=100)

        # Row 1: RGB
        show_rgb(
            axes[0, 0],
            img1_rgb_raw,
            "RGB Sintel\nMpiSintel input_mode='rgb'",
        )
        plot_hex_rgb(
            axes[0, 1],
            rgb_hex,
            eye,
            "RGB hex Sintel\nintermediate BoxEye hexals",
        )
        show_rgb(
            axes[0, 2],
            img1_rgb_raft,
            f"RGB RAFT input\nfrom MpiSintel(..., flyvis_hex=True)\n{raft_img_shape}",
        )

        # Row 2: luminance
        show_rgb(
            axes[1, 0],
            img1_lum_raw,
            "Lum Sintel\nMpiSintel input_mode='lum'",
        )
        plot_hex_lum(
            axes[1, 1],
            lum_hex,
            eye,
            "Lum hex Sintel\nintermediate BoxEye hexals",
        )
        show_rgb(
            axes[1, 2],
            img1_lum_raft,
            f"Lum RAFT input\nfrom MpiSintel(..., flyvis_hex=True)\n{raft_img_shape}",
        )

        # Row 3: optical flow
        show_flow(
            axes[2, 0],
            flow_raw,
            "Optic flow Sintel\nraw target [2,H,W]",
        )
        plot_hex_flow(
            axes[2, 1],
            flow_hex,
            eye,
            "Optic flow hex Sintel\nintermediate BoxEye hexals",
        )
        show_flow(
            axes[2, 2],
            flow_raft_masked,
            f"Optic flow RAFT target\nfrom MpiSintel(..., flyvis_hex=True)\n{raft_flow_shape}",
        )

        if args.use_train_aug:
            aug_text = "training-like random crop/scale/flip"
        else:
            aug_text = "no augmentation, full Sintel frame"

        fig.suptitle(
            f"RAFT FlyVis-hex dataloader check | scene={args.scene} | "
            f"pair={sample_idx}/{n_total - 1} | dstype={args.dstype}\n"
            f"Columns: raw Sintel / intermediate hexals / actual dataloader output fed to RAFT | "
            f"{aug_text}",
            fontsize=12,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        video_frames.append(make_video_frame(fig))
        plt.close(fig)

        print(f"Rendered pair {sample_idx}")

    # -----------------------------------------------------------------
    # Save infinite-loop GIF
    # -----------------------------------------------------------------
    imageio.mimsave(
        out_path,
        video_frames,
        fps=args.fps,
        loop=0,
    )

    print("\nSaved GIF to:")
    print(out_path)

    # -----------------------------------------------------------------
    # Optional MP4
    # -----------------------------------------------------------------
    if args.save_mp4:
        mp4_path = REPO_ROOT / args.mp4_out
        mp4_path.parent.mkdir(parents=True, exist_ok=True)

        imageio.mimsave(
            mp4_path,
            video_frames,
            fps=args.fps,
            macro_block_size=16,
        )

        print("\nSaved MP4 to:")
        print(mp4_path)


if __name__ == "__main__":
    main()