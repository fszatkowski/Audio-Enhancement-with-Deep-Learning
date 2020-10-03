from dataclasses import dataclass

from common.metadata import Metadata


@dataclass
class SEGANMetadata(Metadata):
    # model parameters
    g_lr: float = 0.0002
    d_lr: float = 0.0002
    l1_alpha: float = 10.0

    g_norm: str = None
    d_norm: str = "bnorm"
    """
    There are n_layers convolutional layers in encoder, first layer takes 1 input channels and outputs init_channels 
    subsequent layers multiplicate number of channels by multiplier
    for decoder in both generator we have n+1 layers with channels in reverse order
    """
    n_layers: int = 7
    init_channels: int = 2
    kernel_size: int = 15
    multiplier: int = 2

    def __post_init__(self):
        if self.input_samples % (self.multiplier ** self.n_layers) != 0:
            raise AttributeError(
                f"Input size not multiplier of {self.multiplier}^{self.n_layers})."
            )
        if (self.target_sr // self.input_sr) != self.multiplier:
            raise AttributeError(
                f"Target sr: {self.target_sr} not equal to {self.multiplier} * {self.input_sr})."
            )
