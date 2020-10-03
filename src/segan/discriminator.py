from typing import Sequence

import torch

from segan.modules import ConvBlock


class Discriminator(torch.nn.Module):
    def __init__(
        self,
        kernel_size: int = 31,
        stride: int = 2,
        feature_maps: Sequence[int] = None,
        linear_units: int = None,
        norm: str = "bnorm",
    ):
        super(Discriminator, self).__init__()
        if feature_maps is None:
            raise ValueError("Missing feature maps for discriminator module.")

        encoder_layers = []
        """ Number of feature maps has to be doubled since we concatenate generated and real inputs during inference"""
        feature_maps = [2 * f for f in feature_maps]
        for i, j in zip(feature_maps[:-1], feature_maps[1:]):
            encoder_layers.append(
                ConvBlock(
                    in_channels=i,
                    out_channels=j,
                    kernel_size=kernel_size,
                    stride=stride,
                    norm_type=norm,
                    activation="lrelu",
                )
            )
        self.encoder = torch.nn.ModuleList(encoder_layers)
        self.conv = torch.nn.Conv2d(
            in_channels=feature_maps[-1],
            out_channels=1,
            stride=(1, 1),
            kernel_size=(2, 1),
        )
        self.dense = torch.nn.Linear(linear_units, 1)
        self.classification = torch.nn.Sigmoid()

    def forward(self, generated: torch.Tensor, real: torch.Tensor):
        x = torch.cat((generated, real), dim=1)
        for layer in self.encoder:
            x = layer(x)
        x = self.conv(x)
        x = self.dense(x)
        x = torch.squeeze(x)
        return self.classification(x)
