import torch
import torch.nn.functional as F

from .hexRenderer import BoxEye
from .baseline_cnn import RegularHexToCartesianMap


class RAFTFlyVisHexInput:
    """FlyVis-style hex preprocessing for RAFT.

    Pipeline:
        image / flow / valid
        -> BoxEye
        -> 721 hexals
        -> RegularHexToCartesianMap
        -> 31x31
        -> pad to 32x32
        -> upsample to output_size x output_size

    Default output:
        image: [3, 64, 64]
        flow:  [2, 64, 64]
        valid: [64, 64]
    """

    def __init__(
        self,
        extent=15,
        kernel_size=13,
        output_size=64,
        device="cpu",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.extent = extent
        self.kernel_size = kernel_size
        self.output_size = output_size

        self.eye = BoxEye(extent=extent, kernel_size=kernel_size)
        self.eye.conv = self.eye.conv.to(self.device)
        self.eye.receptor_centers = self.eye.receptor_centers.to(self.device)
        self.eye.min_frame_size = self.eye.min_frame_size.to(self.device)

        self.to_cartesian = RegularHexToCartesianMap(extent=extent).to(self.device)

    def _hexals_to_cart32(self, hexals):
        """Convert hexals to padded 32x32 cartesian map.

        Args:
            hexals:
                [1, 1, 1, 721]
                or
                [1, 1, C, 721]

        Returns:
            cart32:
                [C, 32, 32]
        """
        cart = self.to_cartesian(hexals)

        if cart.ndim == 4:
            # [1, 1, Hc, Wc] -> [1, Hc, Wc]
            cart = cart[0, 0][None]

        elif cart.ndim == 5:
            # [1, 1, C, Hc, Wc] -> [C, Hc, Wc]
            cart = cart[0, 0]

        else:
            raise RuntimeError(f"Unexpected cart shape: {cart.shape}")

        # Usually 31x31 -> 32x32.
        cart = F.pad(cart, (0, 1, 0, 1))

        return cart

    def _resize_image_like(self, x):
        """Resize image-like tensor [C,32,32] to [C,output_size,output_size]."""
        if self.output_size == 32:
            return x

        x = x[None]  # [1, C, 32, 32]
        x = F.interpolate(
            x,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )

        return x[0]

    def _resize_flow_like(self, x):
        """Resize flow tensor [2,32,32] to [2,output_size,output_size].

        This only densifies the spatial grid. It does not rescale flow
        magnitudes, because the current target still uses the sampled Sintel
        flow-value convention.

        Do not multiply by output_size / 32 here unless you also redefine the
        flow target in units of the RAFT input grid.
        """
        if self.output_size == 32:
            return x

        x = x[None]  # [1, 2, 32, 32]
        x = F.interpolate(
            x,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )

        return x[0]

    def _resize_valid_like(self, x):
        """Resize valid mask [32,32] to [output_size,output_size]."""
        if self.output_size == 32:
            return x

        x = x[None, None].float()  # [1, 1, 32, 32]
        x = F.interpolate(
            x,
            size=(self.output_size, self.output_size),
            mode="nearest",
        )

        return x[0, 0]

    def _sample_channels(self, x):
        """Sample each channel independently with BoxEye.

        Args:
            x:
                [C, H, W]

        Returns:
            hexals:
                [1, 1, C, 721]
        """
        x = x.to(self.device, non_blocking=True)

        out = []
        for c in range(x.shape[0]):
            seq = x[c][None, None]  # [1, 1, H, W]
            h = self.eye(seq, ftype="mean", hex_sample=True)  # [1, 1, 1, 721]
            out.append(h)

        return torch.cat(out, dim=2)  # [1, 1, C, 721]

    def _batch_sample_channels(self, x):
        """Sample each channel independently with BoxEye.

        Args:
            x: [B, C, H, W]

        Returns:
            hexals: [B, 1, C, 721]
        """
        x = x.to(self.device, non_blocking=True)

        out = []
        for c in range(x.shape[1]):
            seq = x[:, c][:, None]  # [B, 1, H, W]
            h = self.eye(seq, ftype="mean", hex_sample=True)  # [B, 1, 1, 721]
            out.append(h)

        return torch.cat(out, dim=2)  # [B, 1, C, 721]


    def _batch_hexals_to_cart32(self, hexals):
        """Convert batch hexals to padded 32x32 cartesian maps.

        Args:
            hexals: [B, 1, C, 721]

        Returns:
            cart32: [B, C, 32, 32]
        """
        cart = self.to_cartesian(hexals)

        if cart.ndim == 4:
            # [B, 1, Hc, Wc] -> [B, 1, Hc, Wc]
            cart = cart[:, 0][:, None]

        elif cart.ndim == 5:
            # [B, 1, C, Hc, Wc] -> [B, C, Hc, Wc]
            cart = cart[:, 0]

        else:
            raise RuntimeError(f"Unexpected cart shape: {cart.shape}")

        cart = F.pad(cart, (0, 1, 0, 1))  # 31x31 -> 32x32
        return cart


    def _batch_resize_image_like(self, x):
        """Resize image-like tensor [B,C,32,32] to [B,C,O,O]."""
        if self.output_size == 32:
            return x

        return F.interpolate(
            x,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )


    def _batch_resize_flow_like(self, x):
        """Resize flow tensor [B,2,32,32] to [B,2,O,O].

        This densifies the target grid but does not rescale flow magnitudes.
        """
        if self.output_size == 32:
            return x

        return F.interpolate(
            x,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )


    def _batch_resize_valid_like(self, x):
        """Resize valid mask [B,32,32] to [B,O,O]."""
        if self.output_size == 32:
            return x

        x = x[:, None].float()
        x = F.interpolate(
            x,
            size=(self.output_size, self.output_size),
            mode="nearest",
        )
        return x[:, 0]


    def batch_image_to_raft_input(self, images, input_mode="rgb"):
        """Convert image batch to RAFT-compatible hex input.

        Args:
            images: [B,3,H,W]
            input_mode: "rgb" or "lum"

        Returns:
            [B,3,output_size,output_size]
        """
        images = images.to(self.device, non_blocking=True)

        if input_mode == "lum":
            # Dataset has already repeated luminance over 3 channels.
            # Sample only one channel, then repeat after hex preprocessing.
            x = images[:, :1]
            hexals = self._batch_sample_channels(x)             # [B,1,1,721]
            image32 = self._batch_hexals_to_cart32(hexals)      # [B,1,32,32]
            image = self._batch_resize_image_like(image32)      # [B,1,O,O]
            return image.repeat(1, 3, 1, 1)

        if input_mode == "rgb":
            hexals = self._batch_sample_channels(images)        # [B,1,3,721]
            image32 = self._batch_hexals_to_cart32(hexals)      # [B,3,32,32]
            image = self._batch_resize_image_like(image32)      # [B,3,O,O]
            return image

        raise ValueError(f"Unknown input_mode: {input_mode}")


    def batch_flow_to_raft_target(self, flows):
        """Convert flow batch to RAFT-compatible hex target.

        Args:
            flows: [B,2,H,W]

        Returns:
            [B,2,output_size,output_size]
        """
        flows = flows.to(self.device, non_blocking=True)

        hexals = self._batch_sample_channels(flows)             # [B,1,2,721]
        flow32 = self._batch_hexals_to_cart32(hexals)           # [B,2,32,32]
        flow_out = self._batch_resize_flow_like(flow32)         # [B,2,O,O]

        return flow_out


    def batch_valid_to_raft_mask(self, valid):
        """Convert valid batch to RAFT-compatible valid mask.

        Args:
            valid: [B,H,W]

        Returns:
            [B,output_size,output_size]
        """
        valid = valid.to(self.device, non_blocking=True).float()

        hexals = self.eye(valid[:, None], ftype="mean", hex_sample=True)  # [B,1,1,721]
        mask32 = self._batch_hexals_to_cart32(hexals)[:, 0]               # [B,32,32]

        mask32 = (mask32 > 0.0).float()
        mask_out = self._batch_resize_valid_like(mask32)

        return mask_out

    def image_to_raft_input(self, img):
        """Convert one Sintel image to RAFT-compatible hex input.

        Args:
            img:
                [3, H, W]

        Returns:
            image_hex:
                [3, output_size, output_size]
        """
        img = img.to(self.device, non_blocking=True)

        hexals = self._sample_channels(img)       # [1, 1, 3, 721]
        image32 = self._hexals_to_cart32(hexals)  # [3, 32, 32]
        image = self._resize_image_like(image32)  # [3, output_size, output_size]

        return image

    def flow_to_raft_target(self, flow):
        """Convert one Sintel flow target to RAFT-compatible hex target.

        Args:
            flow:
                [2, H, W]

        Returns:
            flow_hex:
                [2, output_size, output_size]
        """
        flow = flow.to(self.device, non_blocking=True)

        hexals = self._sample_channels(flow)      # [1, 1, 2, 721]
        flow32 = self._hexals_to_cart32(hexals)   # [2, 32, 32]
        flow_out = self._resize_flow_like(flow32) # [2, output_size, output_size]

        return flow_out

    def valid_to_raft_mask(self, valid):
        """Create valid mask in the same output space.

        Args:
            valid:
                [H, W]

        Returns:
            valid_hex:
                [output_size, output_size]
        """
        valid = valid.to(self.device, non_blocking=True).float()

        hexals = self.eye(valid[None, None], ftype="mean", hex_sample=True)
        mask32 = self._hexals_to_cart32(hexals)[0]  # [32, 32]

        # Keep only positions that correspond to valid sampled hexals.
        # Empty cartesian embedding locations and padding are invalid.
        mask32 = (mask32 > 0.0).float()
        mask_out = self._resize_valid_like(mask32)

        return mask_out