# -*- coding: utf-8 -*-
"""
Device-safe BoxEye for RAFT FlyVis-hex preprocessing.

This version is intentionally minimal. It is designed for the RAFT-Sintel
hex preprocessing pipeline:

    image / flow / valid
    -> BoxEye
    -> 721 hexals
    -> RegularHexToCartesianMap
    -> RAFT input

It avoids relying on flyvis.device or torch default CUDA device.
"""

from itertools import product
from typing import Iterator, Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as ttf
from torch import nn

__all__ = ["BoxEye"]


def median(sequence: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Median filter for [samples, frames, H, W].

    Returns:
        out: [samples * frames, 1, H, W]
    """
    samples, frames, height, width = sequence.shape

    x = sequence.reshape(samples * frames, 1, height, width)

    pad = (kernel_size - 1) // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")

    patches = F.unfold(x, kernel_size=kernel_size)
    patches = patches.view(samples * frames, kernel_size * kernel_size, height, width)

    out = patches.median(dim=1).values[:, None]

    return out


class BoxEye:
    """Box filter to produce hexals matching a regular photoreceptor array.

    Args:
        extent:
            Radius, in number of receptors, of the hexagonal array.
            extent=15 gives 721 hexals.

        kernel_size:
            Photon collection radius in pixels.

    Expected input:
        sequence: [samples, frames, height, width]

    Output with hex_sample=True:
        [samples, frames, 1, hexals]
    """

    def __init__(self, extent: int = 15, kernel_size: int = 13):
        self.extent = extent
        self.kernel_size = kernel_size

        self.receptor_centers = torch.tensor(
            list(self._receptor_centers()),
            dtype=torch.long,
        )

        self.hexals = len(self.receptor_centers)

        self.min_frame_size = (
            self.receptor_centers.max(dim=0).values
            - self.receptor_centers.min(dim=0).values
            + 1
        )

        self._set_filter()

        pad = (self.kernel_size - 1) / 2
        self.pad = (
            int(np.ceil(pad)),
            int(np.floor(pad)),
            int(np.ceil(pad)),
            int(np.floor(pad)),
        )

    def _receptor_centers(self) -> Iterator[Tuple[float, float]]:
        """Generate y, x receptor center coordinates."""
        n = self.extent
        d = self.kernel_size

        for u in range(-n, n + 1):
            v_min = max(-n, -n - u)
            v_max = min(n, n - u)

            for v in range(v_min, v_max + 1):
                y = d * (u + v / 2)
                x = d * v
                yield y, x

    def _set_filter(self) -> None:
        """Set up a non-trainable box filter."""
        self.conv = nn.Conv2d(
            1,
            1,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
            bias=True,
        )

        with torch.no_grad():
            self.conv.weight.fill_(1.0)
            self.conv.bias.fill_(0.0)

        self.conv.weight.requires_grad = False
        self.conv.bias.requires_grad = False

    def _move_to_device(self, device: torch.device) -> None:
        """Move internal tensors/modules to the same device as the input."""
        self.conv = self.conv.to(device)
        self.receptor_centers = self.receptor_centers.to(device)
        self.min_frame_size = self.min_frame_size.to(device)

    def __call__(
        self,
        sequence: torch.Tensor,
        ftype: Literal["mean", "sum", "median"] = "mean",
        hex_sample: bool = True,
    ) -> torch.Tensor:
        """Apply box filter to all frames in a sequence.

        Args:
            sequence:
                Cartesian movie sequence with shape [samples, frames, H, W].

            ftype:
                "mean", "sum", or "median".

            hex_sample:
                If True, sample the filtered image at receptor centers.
                If False, return the filtered Cartesian sequence.

        Returns:
            If hex_sample=True:
                [samples, frames, 1, hexals]

            If hex_sample=False:
                [samples, frames, H, W]
        """
        if not isinstance(sequence, torch.Tensor):
            sequence = torch.as_tensor(sequence, dtype=torch.float32)

        sequence = sequence.float()

        if sequence.ndim != 4:
            raise ValueError(
                f"BoxEye expects sequence with shape [samples, frames, H, W], "
                f"but got {tuple(sequence.shape)}"
            )

        samples, frames, height, width = sequence.shape

        device = sequence.device
        self._move_to_device(device)

        current_frame_size = torch.tensor(
            [height, width],
            dtype=self.min_frame_size.dtype,
            device=device,
        )

        if (self.min_frame_size > current_frame_size).any():
            sequence = ttf.resize(
                sequence,
                self.min_frame_size.detach().cpu().tolist(),
            )
            height, width = sequence.shape[2:]

        def _convolve() -> torch.Tensor:
            """Convolve all samples and frames in one batched call."""
            padded = F.pad(sequence, self.pad)
            x = padded.reshape(samples * frames, 1, padded.shape[-2], padded.shape[-1])
            return self.conv(x)

        if ftype == "mean":
            out = _convolve() / float(self.kernel_size**2)

        elif ftype == "sum":
            out = _convolve()

        elif ftype == "median":
            out = median(sequence, self.kernel_size)

        else:
            raise ValueError(
                f"ftype must be 'mean', 'sum', or 'median', but got {ftype}."
            )

        if hex_sample:
            out = self.hex_render(out)
            return out.reshape(samples, frames, 1, -1).contiguous()

        return out.reshape(samples, frames, height, width).contiguous()

    def hex_render(self, sequence: torch.Tensor) -> torch.Tensor:
        """Sample receptor locations from Cartesian frames.

        Args:
            sequence:
                [samples * frames, 1, H, W]

        Returns:
            [samples * frames, 1, 1, hexals]
        """
        if sequence.ndim != 4:
            raise ValueError(
                f"hex_render expects [samples, frames/channels, H, W], "
                f"but got {tuple(sequence.shape)}"
            )

        h, w = sequence.shape[2:]
        device = sequence.device

        self._move_to_device(device)

        current_frame_size = torch.tensor(
            [h, w],
            dtype=self.min_frame_size.dtype,
            device=device,
        )

        if (self.min_frame_size > current_frame_size).any():
            sequence = ttf.resize(
                sequence,
                self.min_frame_size.detach().cpu().tolist(),
            )
            h, w = sequence.shape[2:]

        center_offset = torch.tensor(
            [h // 2, w // 2],
            dtype=self.receptor_centers.dtype,
            device=device,
        )

        centers = self.receptor_centers + center_offset

        y = centers[:, 0].long()
        x = centers[:, 1].long()

        out = sequence[:, :, y, x].clone()

        return out.view(*sequence.shape[:2], 1, -1).contiguous()