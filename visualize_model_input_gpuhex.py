# -*- coding: utf-8 -*-
"""
Visualize exactly what RAFT receives in the GPU-HEX pipeline.

This script shows the tensors after:

    MpiSintel
    -> optional RAFT train augmentation/crop
    -> RAFTFlyVisHexInput
    -> optional train noise
    -> RAFT input tensors

So the displayed image1/image2/flow/valid are the actual tensors used by the model,
not the original Sintel image.
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
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_item_deterministic(dataset, idx, seed=12345):
    set_seed(seed)
    return dataset[idx]


# ---------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------
def image_tensor_to_numpy(img):
    """
    img: [3,H,W], torch, expected 0..255.
    returns [H,W,3] in 0..1.
    """
    img = img.detach().cpu().float()

    if img.max() > 1.5:
        img = img.clamp(0, 255) / 255.0
    else:
        img = img.clamp(0, 1)

    return img.permute(1, 2, 0).numpy()


def gray_tensor_to_numpy(x):
    x = x.detach().cpu().float()

    while x.ndim > 2:
        x = x.squeeze(0)

    return x.numpy()


def flow_tensor_to_rgb(flow):
    """
    flow: [2,H,W], torch.
    returns [H,W,3] uint8.
    """
    flow_np = flow.detach().cpu().float().permute(1, 2, 0).numpy()
    return flow_viz.flow_to_image(flow_np).astype(np.uint8)


def show_img(ax, img, title):
    ax.imshow(image_tensor_to_numpy(img), interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")


def show_flow(ax, flow, title):
    ax.imshow(flow_tensor_to_rgb(flow), interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")


def show_mask(ax, mask, title):
    ax.imshow(
        gray_tensor_to_numpy(mask),
        cmap="gray",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    ax.set_title(title)
    ax.axis("off")


def print_tensor_stats(name, x):
    x_cpu = x.detach().cpu().float()
    print(
        f"{name:12s}",
        "shape=", tuple(x_cpu.shape),
        "dtype=", x_cpu.dtype,
        "min=", float(x_cpu.min()),
        "max=", float(x_cpu.max()),
        "mean=", float(x_cpu.mean()),
        "std=", float(x_cpu.std()),
    )


def print_flow_stats(flow, valid=None, name="flow"):
    flow = flow.detach().cpu().float()

    mag = torch.sqrt((flow ** 2).sum(dim=0))

    if valid is not None:
        valid = valid.detach().cpu().float() >= 0.5
        mag_valid = mag[valid]
    else:
        mag_valid = mag.reshape(-1)

    print(f"\n{name} magnitude")
    print("mean |v|:", float(mag_valid.mean()))
    print("max  |v|:", float(mag_valid.max()))
    print("u min/max:", float(flow[0].min()), float(flow[0].max()))
    print("v min/max:", float(flow[1].min()), float(flow[1].max()))


# ---------------------------------------------------------------------
# Same preprocessing block used before RAFT
# ---------------------------------------------------------------------
@torch.no_grad()
def make_model_tensors(
    img1,
    img2,
    flow,
    valid,
    input_mode,
    output_size,
    device,
    add_noise=False,
    noise_seed=12345,
):
    """
    Convert one MpiSintel sample into exactly the tensors RAFT sees.

    Returns:
        model_img1: [3,O,O]
        model_img2: [3,O,O]
        model_flow: [2,O,O]
        model_valid: [O,O]
    """
    preprocessor = RAFTFlyVisHexInput(
        extent=15,
        kernel_size=13,
        output_size=output_size,
        device=device,
    )

    img1_b = img1[None].to(device, non_blocking=True)
    img2_b = img2[None].to(device, non_blocking=True)
    flow_b = flow[None].to(device, non_blocking=True)
    valid_b = valid[None].to(device, non_blocking=True)

    model_img1 = preprocessor.batch_image_to_raft_input(
        img1_b,
        input_mode=input_mode,
    )
    model_img2 = preprocessor.batch_image_to_raft_input(
        img2_b,
        input_mode=input_mode,
    )
    model_flow = preprocessor.batch_flow_to_raft_target(flow_b, flow_units="raft")
    model_valid = preprocessor.batch_valid_to_raft_mask(valid_b)

    if add_noise:
        # Mirrors the train.py logic if --add_noise is used.
        # Noise is applied after HEX preprocessing and before RAFT.
        torch.manual_seed(noise_seed)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(noise_seed)

        stdv = np.random.uniform(0.0, 5.0)
        model_img1 = (model_img1 + stdv * torch.randn_like(model_img1)).clamp(0.0, 255.0)
        model_img2 = (model_img2 + stdv * torch.randn_like(model_img2)).clamp(0.0, 255.0)

        print(f"\nApplied train-style add_noise with stdv={stdv:.4f}")

    return {
        "image1": model_img1[0],
        "image2": model_img2[0],
        "flow": model_flow[0],
        "valid": model_valid[0],
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scene", type=str, default="alley_1")
    parser.add_argument("--dstype", type=str, default="clean", choices=["clean", "final"])
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--input_mode", type=str, default="lum", choices=["rgb", "lum"])

    parser.add_argument(
        "--use_train_aug",
        action="store_true",
        help="Use the same RAFT crop/scale/flip augmentation setup as training.",
    )
    parser.add_argument(
        "--add_noise",
        action="store_true",
        help="Apply the train.py add_noise block after HEX preprocessing.",
    )

    parser.add_argument("--output_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=12345)

    parser.add_argument(
        "--out",
        type=str,
        default="debug_hex/model_input_gpuhex.png",
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

    dataset = MpiSintel(
        aug_params=aug_params,
        split="training",
        dstype=args.dstype,
        scenes=[args.scene],
        input_mode=args.input_mode,
    )

    img1_raw, img2_raw, flow_raw, valid_raw = get_item_deterministic(
        dataset,
        args.sample_idx,
        seed=args.seed + args.sample_idx,
    )

    model_tensors = make_model_tensors(
        img1=img1_raw,
        img2=img2_raw,
        flow=flow_raw,
        valid=valid_raw,
        input_mode=args.input_mode,
        output_size=args.output_size,
        device=args.device,
        add_noise=args.add_noise,
        noise_seed=args.seed + args.sample_idx,
    )

    image1 = model_tensors["image1"]
    image2 = model_tensors["image2"]
    flow = model_tensors["flow"]
    valid = model_tensors["valid"]

    # -----------------------------------------------------------------
    # Print sanity info
    # -----------------------------------------------------------------
    print("\nRaw MpiSintel output, before GPU-HEX")
    print_tensor_stats("raw image1", img1_raw)
    print_tensor_stats("raw image2", img2_raw)
    print_tensor_stats("raw flow", flow_raw)
    print_tensor_stats("raw valid", valid_raw)
    print_flow_stats(flow_raw, valid_raw, name="raw flow")

    print("\nActual RAFT model tensors, after GPU-HEX")
    print_tensor_stats("model image1", image1)
    print_tensor_stats("model image2", image2)
    print_tensor_stats("model flow", flow)
    print_tensor_stats("model valid", valid)
    print_flow_stats(flow, valid, name="model flow")

    print("\nValid pixels")
    print("model valid:", int((valid >= 0.5).sum().item()), "/", valid.numel())

    # -----------------------------------------------------------------
    # Plot exactly what the model sees
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=120)

    title_suffix = (
        f"scene={args.scene}, sample={args.sample_idx}, dstype={args.dstype}, "
        f"input_mode={args.input_mode}, train_aug={args.use_train_aug}, "
        f"add_noise={args.add_noise}, output_size={args.output_size}"
    )

    show_img(
        axes[0, 0],
        image1,
        "MODEL INPUT image1\npost HEX",
    )
    show_img(
        axes[0, 1],
        image2,
        "MODEL INPUT image2\npost HEX",
    )

    diff = (image2.float() - image1.float()).abs().mean(dim=0)
    axes[0, 2].imshow(
        diff.detach().cpu().numpy(),
        cmap="gray",
        interpolation="nearest",
    )
    axes[0, 2].set_title("MODEL INPUT |image2 - image1|\nmean over channels")
    axes[0, 2].axis("off")

    show_flow(
        axes[1, 0],
        flow,
        "MODEL TARGET flow\npost HEX",
    )
    show_mask(
        axes[1, 1],
        valid,
        "MODEL VALID mask\npost HEX",
    )

    flow_mag = torch.sqrt((flow.detach().cpu().float() ** 2).sum(dim=0))
    axes[1, 2].imshow(
        flow_mag.numpy(),
        cmap="magma",
        interpolation="nearest",
    )
    axes[1, 2].set_title("MODEL TARGET flow magnitude")
    axes[1, 2].axis("off")

    fig.suptitle(
        "Exact tensors given to RAFT after GPU-HEX preprocessing\n" + title_suffix,
        fontsize=12,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path)

    print("\nSaved plot to:")
    print(out_path)

    plt.show()


if __name__ == "__main__":
    main()
    H, W = flow_raw.shape[-2:]
    O = flow.shape[-1]

    raw_mag = torch.sqrt((flow_raw.detach().cpu().float() ** 2).sum(dim=0))
    model_mag = torch.sqrt((flow.detach().cpu().float() ** 2).sum(dim=0))

    raw_valid = valid_raw.detach().cpu().float() >= 0.5
    model_valid = valid.detach().cpu().float() >= 0.5

    print("\nCoordinate-scale sanity")
    print("raw H,W:", H, W)
    print("model O:", O)
    print("expected u scale if RAFT-grid:", O / W)
    print("expected v scale if RAFT-grid:", O / H)

    print("\nRaw flow")
    print("mean |flow|:", float(raw_mag[raw_valid].mean()))
    print("max  |flow|:", float(raw_mag[raw_valid].max()))
    print("u min/max:", float(flow_raw[0].min()), float(flow_raw[0].max()))
    print("v min/max:", float(flow_raw[1].min()), float(flow_raw[1].max()))

    print("\nModel target flow")
    print("mean |flow|:", float(model_mag[model_valid].mean()))
    print("max  |flow|:", float(model_mag[model_valid].max()))
    print("u min/max:", float(flow[0].min()), float(flow[0].max()))
    print("v min/max:", float(flow[1].min()), float(flow[1].max()))