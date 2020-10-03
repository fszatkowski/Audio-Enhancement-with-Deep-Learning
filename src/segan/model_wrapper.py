import os
from typing import Any, Sequence, Tuple

import torch

from common.model_wrapper import ModelWrapper
from segan.main_module import SEGANModule
from segan.metadata import SEGANMetadata


class SEGANWrapper(ModelWrapper):
    def __init__(self, metadata: SEGANMetadata, loss: Any = torch.nn.MSELoss()):
        super(SEGANWrapper, self).__init__(
            net=SEGANModule(
                kernel_size=metadata.kernel_size,
                stride=metadata.multiplier,
                n_layers=metadata.n_layers,
                init_channels=metadata.init_channels,
                g_norm=metadata.g_norm,
                d_norm=metadata.d_norm,
                d_linear_units=metadata.input_samples
                // (metadata.multiplier ** metadata.n_layers),
            ),
            metadata=metadata,
            loss=loss,
        )

        self.prepare_for_gpu()
        self.g_optimizer = torch.optim.Adam(
            self.net.generator.parameters(), metadata.g_lr
        )
        self.d_optimizer = torch.optim.Adam(
            self.net.discriminator.parameters(), metadata.d_lr
        )
        self.l1_weight = metadata.l1_alpha

    def train_step(
        self, batch: Sequence[torch.Tensor]
    ) -> Tuple[float, float, float, float]:
        inputs, targets, dist_samples = batch
        labels = torch.ones((inputs.shape[0]))
        if torch.cuda.is_available():
            labels = labels.cuda()
        true_targets = labels * 1
        false_targets = labels * 0

        self.d_optimizer.zero_grad()

        # D real update
        d_real_output = self.net.discriminator(dist_samples, targets)
        d_real_loss = self.loss(d_real_output, true_targets)
        d_real_loss.backward()

        # D fake update
        g_output = self.net.generator(inputs)
        d_fake_output = self.net.discriminator(g_output.detach(), targets)
        d_fake_loss = self.loss(d_fake_output, false_targets)
        d_fake_loss.backward()

        self.d_optimizer.step()

        self.g_optimizer.zero_grad()

        # G adversarial update
        g_fake_result = self.net.discriminator(g_output, targets)
        g_adversarial_loss = self.loss(g_fake_result, true_targets)

        # G l1 update
        g_l1_loss = self.l1_weight * self.loss(g_output, targets)

        g_loss = g_adversarial_loss + g_l1_loss
        g_loss.backward()
        self.g_optimizer.step()

        return (
            d_real_loss.item(),
            d_fake_loss.item(),
            g_adversarial_loss.item(),
            g_l1_loss.item(),
        )

    def compute_mse_loss(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.net(inputs)
        return self.loss(outputs, targets)
