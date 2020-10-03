import torch


class AutoencoderModule(torch.nn.Module):
    """ It is expected that input shape will be: (batches, 1, audio_channels, samples), where:
        samples size is multiple of 2^num_layers"""

    def __init__(
        self,
        num_layers: int,
        channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        activation: str = "relu",
        norm: str = "none",
    ):
        super(AutoencoderModule, self).__init__()
        encoder_layers = [
            DownsamplingModule(
                1,
                channels,
                kernel_size=kernel_size,
                stride=stride,
                activation=activation,
                norm=norm,
            )
        ] + [
            DownsamplingModule(
                int(channels * 2 ** i),
                int(channels * 2 ** (i + 1)),
                kernel_size,
                stride=stride,
            )
            for i in range(0, num_layers - 1)
        ]
        self.encoder = torch.nn.ModuleList(encoder_layers)
        decoder_layers = [
            UpsamplingModule(
                int(channels * 2 ** (num_layers - i - 1)),
                int(channels * 2 ** (num_layers - i - 2)),
                kernel_size=kernel_size,
                stride=stride,
                activation=activation,
                norm=norm,
            )
            for i in range(0, num_layers)
        ]
        self.decoder = torch.nn.ModuleList(decoder_layers)
        self.output = OutputModule(
            input_channels=channels // 2, upsample_kernel_size=kernel_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        for layer in self.encoder:
            skip_connections.append(x)
            x = layer(x)
        for layer, skip in zip(self.decoder, skip_connections[::-1]):
            x = layer(x)
            x = x + skip
        return self.output(x)


class DownsamplingModule(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 2,
        activation: str = "relu",
        norm: str = "none",
    ):
        super(DownsamplingModule, self).__init__()
        pad_size = int((kernel_size - 1) / 2)

        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, 1),
            padding=(0, pad_size),
        )
        self.pool = torch.nn.MaxPool2d((1, stride), stride=(1, stride))
        self.activation = _get_activation(activation)
        if norm == "batch_norm":
            self.norm = torch.nn.BatchNorm2d(out_channels)
        elif norm == "none":
            self.norm = None
        else:
            raise ValueError(f"Unknown norm type: {norm}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = self.activation(x)
        if self.norm is not None:
            return self.norm(x)
        else:
            return x


class UpsamplingModule(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 2,
        activation: str = "relu",
        norm: str = "none",
    ):
        super(UpsamplingModule, self).__init__()

        pad = max(0, (stride - kernel_size) // -2)
        self.skip_last = kernel_size % 2
        self.conv = torch.nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, pad),
        )
        self.activation = _get_activation(activation)
        if norm == "batch_norm":
            self.norm = torch.nn.BatchNorm2d(out_channels)
        elif norm == "none":
            self.norm = None
        else:
            raise ValueError(f"Unknown norm type: {norm}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.skip_last:
            x = x[:, :, :, :-1]
        x = self.activation(x)
        if self.norm is not None:
            return self.norm(x)
        else:
            return x


class OutputModule(torch.nn.Module):
    def __init__(self, input_channels: int, upsample_kernel_size: int):
        super(OutputModule, self).__init__()

        self.upsample = UpsamplingModule(
            input_channels, input_channels, kernel_size=upsample_kernel_size, stride=2
        )
        self.conv = torch.nn.Conv2d(
            input_channels, 1, kernel_size=(1, 1), stride=(1, 1)
        )
        self.tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        return self.tanh(x)


class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def _get_activation(name: str):
    if name == "relu":
        return torch.nn.ReLU()
    elif name == "elu":
        return torch.nn.ELU()
    elif name == "prelu":
        return torch.nn.PReLU()
    elif name == "gelu":
        return torch.nn.GELU()
    elif name == "swish":
        return Swish()
    else:
        raise ValueError(f"Unknown acitvation: {name}")
