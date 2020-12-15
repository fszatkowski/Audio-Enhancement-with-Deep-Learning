from typing import Any, Sequence

import torch

from common.model_wrapper import ModelWrapper
from wavenet.metadata import WaveNetMetadata
from wavenet.modules import WaveNetModule


class WaveNetWrapper(ModelWrapper):
    def __init__(self, metadata: WaveNetMetadata, loss: Any = torch.nn.MSELoss()):
        super(WaveNetWrapper, self).__init__(
            net=WaveNetModule(
                blocks_per_stack=metadata.stack_layers,
                stack_size=metadata.stack_size,
                input_kernel_size=metadata.input_kernel_size,
                res_channels=metadata.residual_channels,
                skip_kernel_size=metadata.skip_kernel_size,
                skip_channels=metadata.skip_channels,
            ),
            metadata=metadata,
            loss=loss,
        )

        self.prepare_for_gpu()
        self.optimizer = torch.optim.AdamW(
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
        targets = targets[:, :, :, 2 * self.net.receptive_fields :]

        outputs = self.net(inputs)
        return self.loss(outputs, targets)
