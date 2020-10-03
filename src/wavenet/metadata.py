from dataclasses import dataclass

from common.metadata import Metadata


@dataclass
class WaveNetMetadata(Metadata):
    # model parameters
    learning_rate: float = 0.001
    stack_size: int = 4
    stack_layers: int = 5

    input_kernel_size: int = 31
    residual_channels: int = 16
    skip_kernel_size: int = 31
    skip_channels: int = 16
