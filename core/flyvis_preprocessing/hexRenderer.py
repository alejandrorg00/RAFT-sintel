# -*- coding: utf-8 -*-
"""
'Transduction' of cartesian pixels to hexals on a regular hexagonal lattice.
"""

from itertools import product
from typing import Iterator, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as ttf
from torch import nn

import flyvis
from flyvis.analysis.visualization.plt_utils import init_plot, rm_spines

from flyvis.datasets.rendering.utils import (
    hex_center_coordinates,
    is_inside_hex,
    median,
    render_bars_cartesian,
    render_gratings_cartesian,
)

__all__ = ["BoxEye", "HexEye"]

# ----- BoxEye -----------------------------------------------------------------


class BoxEye:
    """BoxFilter to produce an array of hexals matching the photoreceptor array.

    Args:
        extent: Radius, in number of receptors, of the hexagonal array.
        kernel_size: Photon collection radius, in pixels.

    Attributes:
        extent (int): Radius, in number of receptors, of the hexagonal array.
        kernel_size (int): Photon collection radius, in pixels.
        receptor_centers (torch.Tensor): Tensor of shape (hexals, 2) containing the y, x
            coordinates of the hexal centers.
        hexals (int): Number of hexals in the array.
        min_frame_size (torch.Tensor): Minimum frame size to contain the hexal array.
        pad (Tuple[int, int, int, int]): Padding to apply to the frame before convolution.
        conv (nn.Conv2d): Convolutional box filter to apply to the frame.
    """

    def __init__(self, extent: int = 15, kernel_size: int = 13):
        self.extent = extent
        self.kernel_size = kernel_size
        self.receptor_centers = torch.tensor(
            [*self._receptor_centers()], dtype=torch.long
        )
        self.hexals = len(self.receptor_centers)
        # The rest of kernel_size distance from outer centers to the border
        # is taken care of by the padding of the convolution object.
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
        """Generate receptor center coordinates.

        Returns:
            Iterator[Tuple[float, float]]: Yields y, x coordinates of receptor centers.
        """
        n = self.extent
        d = self.kernel_size
        for u in range(-n, n + 1):
            v_min = max(-n, -n - u)
            v_max = min(n, n - u)
            for v in range(v_min, v_max + 1):
                # y = -d * v
                # x = 2 / np.sqrt(3) * d * (u + v/2)
                y = d * (
                    u + v / 2
                )  # - d * v # either must be negative or origin must be upper
                x = d * v  # 2 / np.sqrt(3) * d * (u + v / 2)
                yield y, x
                # xs.append()
                # ys.append()

    def _set_filter(self) -> None:
        """Set up the convolutional filter for the box kernel."""
        self.conv = nn.Conv2d(1, 1, kernel_size=self.kernel_size, stride=1, padding=0)
        self.conv.weight.data /= self.conv.weight.data
        self.conv.bias.data.fill_(0)  # if not self.requires_grad else None
        self.conv.weight.requires_grad = False  # self.requires_grad
        self.conv.bias.requires_grad = False  # self.requires_grad

    def __call__(
        self,
        sequence: torch.Tensor,
        ftype: Literal["mean", "sum", "median"] = "mean",
        hex_sample: bool = True,
    ) -> torch.Tensor:
        """Apply a box kernel to all frames in a sequence.

        Args:
            sequence: Cartesian movie sequences of shape (samples, frames, height, width).
            ftype: Filter type.
            hex_sample: If False, returns filtered cartesian sequences.

        Returns:
            torch.Tensor: Shape (samples, frames, 1, hexals) if hex_sample is True,
                otherwise (samples, frames, height, width).
        """
        samples, frames, height, width = sequence.shape

        if not isinstance(sequence, torch.Tensor):
            # auto-moving to GPU in case default tensor is cuda but passed
            # sequence is not, for convenience
            sequence = torch.tensor(sequence, dtype=torch.float32, device=flyvis.device)
        device = sequence.device
        self.conv = self.conv.to(device)
        min_frame_size = self.min_frame_size.to(device)
        if (self.min_frame_size > torch.tensor([height, width])).any():
            # to rescale to the minimum frame size
            sequence = ttf.resize(sequence, self.min_frame_size.tolist())
            height, width = sequence.shape[2:]

        def _convolve():
            # convole each sample sequentially to avoid gpu memory issues
            def conv(x):
                return self.conv(x.unsqueeze(1))

            return torch.cat(
                tuple(map(conv, torch.unbind(F.pad(sequence, self.pad), dim=0))), dim=0
            )

        if ftype == "mean":
            out = _convolve() / self.kernel_size**2
        elif ftype == "sum":
            out = _convolve()
        elif ftype == "median":
            out = median(sequence, self.kernel_size)
        else:
            raise ValueError("ftype must be 'sum', 'mean', or 'median." f"Is {ftype}.")

        if hex_sample is True:
            return self.hex_render(out).reshape(samples, frames, 1, -1)

        return out.reshape(samples, frames, height, width)

    def hex_render(self, sequence: torch.Tensor) -> torch.Tensor:
        """Sample receptor locations from a sequence of cartesian frames.

        Args:
            sequence: Cartesian movie sequences of shape (samples, frames, height, width).

        Returns:
            torch.Tensor: Shape (samples, frames, 1, hexals).

        Note:
            Resizes the sequence to the minimum frame size if necessary.
        """
        h, w = sequence.shape[2:]
        device = sequence.device

        min_frame_size = self.min_frame_size.to(device)
        receptor_centers = self.receptor_centers.to(device)

        if (min_frame_size > torch.tensor([h, w], device=device)).any():
            sequence = ttf.resize(sequence, min_frame_size.tolist())
            h, w = sequence.shape[2:]

        c = receptor_centers + torch.tensor([h // 2, w // 2], device=device)
        out = sequence[:, :, c[:, 0], c[:, 1]].clone()

        return out.view(*sequence.shape[:2], 1, -1).contiguous()