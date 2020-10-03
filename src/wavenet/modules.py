from typing import Tuple

import numpy as np
import torch


class WaveNetModule(torch.nn.Module):
    def __init__(
        self,
        blocks_per_stack: int,
        stack_size: int,
        input_kernel_size: int,
        res_channels: int,
        skip_kernel_size: int,
        skip_channels: int,
    ):
        super(WaveNetModule, self).__init__()

        self.receptive_fields = int(
            np.sum([2 ** i for i in range(0, blocks_per_stack)] * stack_size)
        )
        self.input_conv = InputConv(
            input_channels=1,
            out_channels=res_channels,
            input_kernel_size=input_kernel_size,
        )
        self.res_stack = ResidualStack(
            blocks_per_stack=blocks_per_stack,
            stack_size=stack_size,
            res_channels=res_channels,
            skip_channels=skip_channels,
            skip_kernel_size=skip_kernel_size,
        )
        self.out = OutputStack(skip_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_size = 2 * (x.size(3) - self.receptive_fields)
        output = self.input_conv(x)
        skip_connections = self.res_stack(output, output_size)
        output = torch.sum(skip_connections, dim=0)
        return self.out(output)


class InputConv(torch.nn.Module):
    def __init__(self, input_channels: int, out_channels: int, input_kernel_size: int):
        super(InputConv, self).__init__()

        if input_kernel_size % 2 == 0:
            raise ValueError("Input kernel size should not be even")

        # padding is used to preserve dimensions
        pad = (input_kernel_size - 1) // 2
        self.conv = torch.nn.Conv2d(
            input_channels,
            out_channels,
            kernel_size=(1, input_kernel_size),
            stride=(1, 1),
            padding=(0, pad),
            bias=True,
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.relu(x)


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        res_channels: int,
        skip_channels: int,
        dilation: int,
        skip_kernel_size: int,
    ):
        super(ResidualBlock, self).__init__()

        self.dilated_tanh = torch.nn.Conv2d(
            res_channels,
            res_channels,
            kernel_size=(1, 2),
            stride=(1, 1),
            dilation=(1, dilation),
            padding=(0, 0),
            bias=True,
        )
        self.dilated_sigmoid = torch.nn.Conv2d(
            res_channels,
            res_channels,
            kernel_size=(1, 2),
            stride=(1, 1),
            dilation=(1, dilation),
            padding=(0, 0),
            bias=True,
        )
        self.conv_res = torch.nn.Conv2d(res_channels, res_channels, (1, 1))
        self.conv_skip = torch.nn.Conv2d(res_channels, skip_channels, (1, 1))

        self.gate_tanh = torch.nn.Tanh()
        self.gate_sigmoid = torch.nn.Sigmoid()

        pad = max(0, (2 - skip_kernel_size) // -2)
        self.skip_last = skip_kernel_size % 2
        self.upsampling_conv = torch.nn.ConvTranspose2d(
            skip_channels,
            skip_channels,
            kernel_size=(1, skip_kernel_size),
            stride=(1, 2),
            padding=(0, pad),
        )

    def forward(
        self, x: torch.Tensor, skip_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_tanh = self.dilated_tanh(x)
        output_sigmoid = self.dilated_sigmoid(x)

        # gate
        gated_tanh = self.gate_tanh(output_tanh)
        gated_sigmoid = self.gate_sigmoid(output_sigmoid)
        gated = gated_tanh * gated_sigmoid

        # skip output network
        skip = self.conv_skip(gated)

        # residual output
        input_cut = x[:, :, :, -skip.size(3) :]
        output = skip + input_cut
        output = self.conv_res(output)

        # modification for upsampling -> convTranspose for output
        skip = self.upsampling_conv(skip)
        if self.skip_last:
            skip = skip[:, :, :, :-1]
        skip = skip[:, :, :, -skip_size:]

        return output, skip


class ResidualStack(torch.nn.Module):
    def __init__(
        self,
        blocks_per_stack: int,
        stack_size: int,
        res_channels: int,
        skip_channels: int,
        skip_kernel_size: int,
    ):
        super(ResidualStack, self).__init__()

        res_blocks = []
        for dilation in stack_size * [2 ** p for p in range(0, blocks_per_stack)]:
            res_block = ResidualBlock(
                res_channels=res_channels,
                skip_channels=skip_channels,
                dilation=dilation,
                skip_kernel_size=skip_kernel_size,
            )
            if torch.cuda.device_count() > 1:
                res_block = torch.nn.DataParallel(res_block)
            if torch.cuda.is_available():
                res_block.cuda()
            res_blocks.append(res_block)
        self.res_blocks = torch.nn.ModuleList(res_blocks)

    def forward(self, x: torch.Tensor, skip_size: int) -> torch.Tensor:
        output = x
        skip_connections = []

        for res_block in self.res_blocks:
            # output is the next input
            output, skip = res_block(output, skip_size)
            skip_connections.append(skip)

        return torch.stack(skip_connections)


class OutputStack(torch.nn.Module):
    def __init__(self, channels: int):
        super(OutputStack, self).__init__()

        self.relu1 = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(channels, channels, (1, 1))
        self.relu2 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(channels, 1, (1, 1))

        self.tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.relu1(x)
        output = self.conv1(output)
        output = self.relu2(output)
        output = self.conv2(output)

        return self.tanh(output)
