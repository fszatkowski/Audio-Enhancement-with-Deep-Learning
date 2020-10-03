from typing import Any, Sequence

import torch

from autoencoder.metadata import AutoencoderMetadata
from autoencoder.modules import AutoencoderModule
from common.model_wrapper import ModelWrapper


class AutoencoderWrapper(ModelWrapper):
    def __init__(self, metadata: AutoencoderMetadata, loss: Any = torch.nn.MSELoss()):
        super(AutoencoderWrapper, self).__init__(
            net=AutoencoderModule(
                num_layers=metadata.num_layers,
                channels=metadata.channels,
                kernel_size=metadata.kernel_size,
                stride=metadata.multiplier,
                activation=metadata.activation,
                norm=metadata.norm,
            ),
            metadata=metadata,
            loss=loss,
        )

        self.prepare_for_gpu()
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=metadata.learning_rate
        )

    def train_step(self, batch: Sequence[torch.Tensor]) -> float:
        self.optimizer.zero_grad()

        loss = self.compute_mse_loss(*batch)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_mse_loss(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.net(inputs)
        return self.loss(outputs, targets)
