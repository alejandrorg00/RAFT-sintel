import torch
import torch.nn.functional as F

from .hexRenderer import BoxEye
from .baseline_cnn import RegularHexToCartesianMap


class RAFTFlyVisHexInput:
    """FlyVis-style hex preprocessing for RAFT.

    This keeps RAFT unchanged. It only changes the dataset output:
    cartesian Sintel image -> BoxEye hexals -> regular cartesian map -> resized RAFT input.
    """

    def __init__(self, extent=15, kernel_size=13, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.eye = BoxEye(extent=extent, kernel_size=kernel_size)
        self.eye.conv = self.eye.conv.to(self.device)
        self.eye.receptor_centers = self.eye.receptor_centers.to(self.device)
        self.eye.min_frame_size = self.eye.min_frame_size.to(self.device)

        self.to_cartesian = RegularHexToCartesianMap(extent=extent).to(self.device)

    def image_to_raft_input(self, img):
        """Convert one Sintel image to a RAFT-compatible hex-rendered image.

        Args:
            img: torch.Tensor [3, H, W], usually in RAFT range 0..255.

        Returns:
            torch.Tensor [3, H, W]
        """
        img = img.to(self.device, non_blocking=True)
        _, H, W = img.shape

        gray = 0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]
        seq = gray[None, None]

        hexals = self.eye(seq, ftype="mean", hex_sample=True)
        cart = self.to_cartesian(hexals)

        if cart.ndim == 3:
            cart = cart[:, None]

        cart = F.pad(cart, (0, 1, 0, 1))
        cart = cart[0, 0]

        return cart[None].repeat(3, 1, 1)

    def flow_to_raft_target(self, flow):
        """Convert one Sintel flow target through the same hex-to-cartesian map.

        Args:
            flow: torch.Tensor [2, H, W]

        Returns:
            torch.Tensor [2, H, W]
        """
        flow = flow.to(self.device, non_blocking=True)
        _, H, W = flow.shape

        out_channels = []
        for c in range(2):
            seq = flow[c][None, None]
            hexals = self.eye(seq, ftype="mean", hex_sample=True)
            cart = self.to_cartesian(hexals)

            if cart.ndim == 3:
                cart = cart[:, None]

            cart = F.interpolate(cart, size=(H, W), mode="nearest")
            out_channels.append(cart[0, 0])

        return torch.stack(out_channels, dim=0)

    def valid_to_raft_mask(self, valid):
        """Create a RAFT valid mask after hex sampling.

        Args:
            valid: torch.Tensor [H, W]

        Returns:
            torch.Tensor [H, W]
        """
        valid = valid.to(self.device, non_blocking=True)
        H, W = valid.shape

        seq = valid.float()[None, None]
        hexals = self.eye(seq, ftype="mean", hex_sample=True)
        cart = self.to_cartesian(hexals)

        if cart.ndim == 3:
            cart = cart[:, None]

        cart = F.interpolate(cart, size=(H, W), mode="nearest")
        mask = cart[0, 0]

        return (mask > 0.99).float()