from dataclasses import dataclass

from common.metadata import Metadata


@dataclass
class AutoencoderMetadata(Metadata):
    # model parameters
    learning_rate: float = 0.001
    num_layers: int = 7
    channels: int = 2
    kernel_size: int = 15
    multiplier: int = 2
    activation: str = "relu"
    norm: str = "none"

    def __post_init__(self):
        if self.input_samples % (self.multiplier ** self.num_layers) != 0:
            raise AttributeError(
                f"Input size not multiplier of {self.multiplier}^{self.num_layers})."
            )
        if (self.target_sr // self.input_sr) != self.multiplier:
            raise AttributeError(
                f"Target sr: {self.target_sr} not equal to {self.multiplier} * {self.input_sr})."
            )
