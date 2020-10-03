from typing import Union

import torch


def build_norm_layer(
    norm_type: str, num_feats: int = None
) -> Union[torch.nn.Module, None]:
    if norm_type == "bnorm":
        return torch.nn.BatchNorm2d(num_feats)
    elif norm_type is None:
        return None
    else:
        raise TypeError("Unrecognized norm type: ", norm_type)


class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 2,
        norm_type: str = None,
        activation: str = "prelu",
    ):
        super().__init__()
        pad = (0, kernel_size // 2)
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=pad,
        )
        self.norm = build_norm_layer(norm_type, out_channels)
        if activation == "prelu":
            self.activation = torch.nn.PReLU(out_channels, init=0)
        elif activation == "lrelu":
            self.activation = torch.nn.LeakyReLU(negative_slope=0.3)
        else:
            raise ValueError(f"{activation} non linearity not supported.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.activation(x)
        return x


class DeconvBlock(torch.nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size=31,
        stride=2,
        norm_type=None,
        activation: str = "prelu",
    ):
        super().__init__()
        pad = max(0, (stride - kernel_size) // -2)
        self.skip_last = kernel_size % 2
        self.deconv = torch.nn.ConvTranspose2d(
            input_channels,
            output_channels,
            (1, kernel_size),
            stride=(1, stride),
            padding=(0, pad),
        )
        self.norm = build_norm_layer(norm_type, output_channels)
        if activation == "prelu":
            self.activation = torch.nn.PReLU(output_channels, init=0)
        elif activation == "tanh":
            self.activation = torch.nn.Tanh()
        else:
            raise ValueError(f"Activation {activation} not supported for DeconvBlock")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.deconv(x)
        if self.skip_last:
            h = h[:, :, :, :-1]
        if self.norm is not None:
            h = self.norm(x)
        h = self.activation(h)
        return h
