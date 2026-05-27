import torch
from torch import nn
from torch.nn.init import kaiming_normal_, constant_
import torchvision.transforms.functional as ttf
import flyvis


def conv_block(
    batchNorm,
    in_planes,
    out_planes,
    kernel_size=3,
    stride=1,
    nonlinearity="ELU",
):
    if nonlinearity is not None:
        if batchNorm:
            return nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    bias=False,
                ),
                nn.BatchNorm2d(out_planes),
                getattr(nn, nonlinearity)(),
            )
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=True,
            ),
            getattr(nn, nonlinearity)(),
        )
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=(kernel_size - 1) // 2,
        bias=True,
    )


class RegularHexToCartesianMap(torch.nn.Module):
    """Translates hexagonal input to a cartesian map to be decoded with
    a cartesian CNN.
    """

    def __init__(self, extent=15):
        super().__init__()

        u, v = flyvis.utils.hex_utils.get_hex_coords(extent)
        u = u - u.min()
        v = v - v.min()

        self.H = int(u.max() + 1)
        self.W = int(v.max() + 1)

        self.register_buffer("u", torch.as_tensor(u, dtype=torch.long))
        self.register_buffer("v", torch.as_tensor(v, dtype=torch.long))

    def forward(self, x):
        """Translates (n_samples, n_frames, ndim, n_hexals)
        to (n_samples, n_frames, *ndim, H, W).
        """
        n_samples, n_frames, in_channels, n_hexals = x.shape

        x_map = torch.zeros(
            [n_samples, n_frames, in_channels, self.H, self.W],
            dtype=x.dtype,
            device=x.device,
        )

        x_map[:, :, :, self.u, self.v] = x
        x_map.squeeze_(dim=2)

        return x_map


class CartesianMapToRegularHex(torch.nn.Module):
    def __init__(self, extent=15):
        super().__init__()

        u, v = flyvis.utils.hex_utils.get_hex_coords(extent)
        u = u - u.min()
        v = v - v.min()

        self.H = int(u.max() + 1)
        self.W = int(v.max() + 1)

        self.register_buffer("u", torch.as_tensor(u, dtype=torch.long))
        self.register_buffer("v", torch.as_tensor(v, dtype=torch.long))

    def forward(self, x):
        n_samples, n_channels = x.shape[:2]
        n_frames = 1

        return x.view(n_samples, n_frames, n_channels, self.H, self.W)[
            :, :, :, self.u, self.v
        ]

class FlowResize(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, H, W):
        if x.shape[-2:] != (H, W):
            return torch.stack(
                (ttf.resize(x[:, 0], (H, W)), ttf.resize(x[:, 1], (H, W))),
                dim=1,
            )
        return x


class VanillaHexCNN(torch.nn.Module):
    def __init__(
        self,
        n_frames=4,
        conv_kernel_sizes=None,
        conv_strides=None,
        batchNorm=True,
    ):
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [1, 3, 3, 3]
        if conv_strides is None:
            conv_strides = [1, 1, 1, 1]
        super().__init__()

        self.to_cartesian = RegularHexToCartesianMap(extent=15)

        self.conv1 = conv_block(
            batchNorm=batchNorm,
            in_planes=n_frames,
            out_planes=64,
            kernel_size=conv_kernel_sizes[0],
            stride=conv_strides[0],
        )
        self.conv2 = conv_block(
            batchNorm=batchNorm,
            in_planes=64,
            out_planes=32,
            kernel_size=conv_kernel_sizes[1],
            stride=conv_strides[1],
        )
        self.conv3 = conv_block(
            batchNorm=batchNorm,
            in_planes=32,
            out_planes=2,
            kernel_size=conv_kernel_sizes[2],
            stride=conv_strides[2],
            nonlinearity=False,
        )
        # self.conv4 = nn.Sequential()
        # # self.conv4 = conv_block_pr(
        # #     batchNorm=batchNorm,
        # #     in_planes=32,
        # #     out_planes=2,
        # #     kernel_size=conv_kernel_sizes[3],
        # #     stride=conv_strides[3],
        # #     nonlinearity=False,
        # # )
        # self.flowresize = nn.Sequential()  # FlowResize()
        self.to_hex = CartesianMapToRegularHex(extent=15)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        # return self.flowresize(
        return self.to_hex(self.conv3(self.conv2(self.conv1(self.to_cartesian(x)))))
        #     *x.shape[-2:]
        # )

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if "weight" in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if "bias" in name]


class VanillaHexCNNBaseline(torch.nn.Module):
    """approx. 400k parameters"""

    def __init__(
        self,
        n_frames=4,
        conv_kernel_sizes=None,
        conv_strides=None,
        batchNorm=True,
        shape=None,
        nonlinearity="ELU",
    ):
        if shape is None:
            shape = [64, 32, 16, 8, 2]
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [1, 3, 3, 3, 5]
        if conv_strides is None:
            conv_strides = [1, 1, 1, 1, 1]
        super().__init__()

        self.to_cartesian = RegularHexToCartesianMap(extent=15)
        conv = []
        conv.append(
            conv_block(
                batchNorm=batchNorm,
                in_planes=n_frames,
                out_planes=shape[0],
                kernel_size=conv_kernel_sizes[0],
                stride=conv_strides[0],
                nonlinearity=nonlinearity,
            )
        )
        in_planes = shape[0]
        for i, out_planes in enumerate(shape[1:-1]):
            conv.append(
                conv_block(
                    batchNorm=batchNorm,
                    in_planes=in_planes,
                    out_planes=out_planes,
                    kernel_size=conv_kernel_sizes[i + 1],
                    stride=conv_strides[i + 1],
                    nonlinearity=nonlinearity,
                )
            )
            in_planes = out_planes
        conv.append(
            conv_block(
                batchNorm=batchNorm,
                in_planes=in_planes,
                out_planes=shape[-1],
                kernel_size=conv_kernel_sizes[-1],
                stride=conv_strides[-1],
                nonlinearity=None,
            )
        )

        self.conv = nn.Sequential(*conv)
        self.to_hex = CartesianMapToRegularHex(extent=15)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        return self.to_hex(self.conv(self.to_cartesian(x)))

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if "weight" in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if "bias" in name]