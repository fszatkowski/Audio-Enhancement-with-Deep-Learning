from typing import Sequence, Tuple

import torch

from segan.modules import ConvBlock, DeconvBlock


class Generator(torch.nn.Module):
    def __init__(
        self,
        kernel_size: int = 31,
        stride: int = 2,
        feature_maps: Sequence[int] = None,
        norm: str = None,
    ):
        super(Generator, self).__init__()
        if feature_maps is None:
            raise ValueError("Missing feature maps for generator module.")

        encoder_layers = []
        for i, j in zip(feature_maps[:-1], feature_maps[1:]):
            encoder_layers.append(
                ConvBlock(
                    in_channels=i,
                    out_channels=j,
                    kernel_size=kernel_size,
                    stride=stride,
                    norm_type=norm,
                )
            )
        self.encoder = torch.nn.ModuleList(encoder_layers)

        decoder_layers = []
        decoder_feature_mappings = [(2 * feature_maps[-1], feature_maps[-1])]
        for n, nn in zip(
            list(reversed(feature_maps))[:-1], list(reversed(feature_maps))[1:]
        ):
            decoder_feature_mappings.append((n + nn, nn))
        for i, o in decoder_feature_mappings[:-1]:
            decoder_layers.append(
                DeconvBlock(
                    input_channels=i,
                    output_channels=o,
                    kernel_size=kernel_size,
                    stride=stride,
                    norm_type=norm,
                )
            )
        decoder_layers.append(
            DeconvBlock(
                decoder_feature_mappings[-1][0],
                decoder_feature_mappings[-1][1],
                kernel_size=kernel_size,
                stride=stride,
                norm_type=norm,
                activation="tanh",
            )
        )
        self.decoder = torch.nn.ModuleList(decoder_layers)

    def forward(self, x: torch.Tensor, z: torch.Tensor = None) -> torch.Tensor:
        skip_connections = []
        for layer in self.encoder:
            skip_connections = [x] + skip_connections
            x = layer(x)
        if z is None:
            z = torch.randn(x.shape)
        if torch.cuda.is_available():
            z = z.cuda()
        x = torch.cat((x, z), dim=1)
        for layer, skip in zip(self.decoder[:-1], skip_connections):
            x = layer(x)
            x = torch.cat((x, skip), dim=1)
        x = self.decoder[-1](x)
        return x
