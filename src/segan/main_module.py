import torch

from segan.discriminator import Discriminator
from segan.generator import Generator


class SEGANModule(torch.nn.Module):
    """ Container for both generator and discriminator """

    def __init__(
        self,
        n_layers: int = 10,
        init_channels: int = 2,
        kernel_size: int = 31,
        stride: int = 2,
        d_linear_units: int = 8,
        g_norm: str = None,
        d_norm: str = None,
    ):
        super(SEGANModule, self).__init__()
        feature_maps = [1] + [init_channels * stride ** i for i in range(n_layers + 1)]
        self.generator = Generator(
            kernel_size=kernel_size,
            stride=stride,
            norm=g_norm,
            feature_maps=feature_maps[:-1],
        )
        self.discriminator = Discriminator(
            kernel_size=kernel_size,
            stride=stride,
            norm=d_norm,
            feature_maps=feature_maps,
            linear_units=d_linear_units,
        )

    def forward(self, x: torch.Tensor):
        return self.generator(x)
