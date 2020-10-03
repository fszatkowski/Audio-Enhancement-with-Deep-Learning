import os
from abc import ABC, abstractmethod
from typing import Any, Sequence

import torch

from common.metadata import Metadata
from constants import MODEL_FILENAME


class ModelWrapper(ABC):
    def __init__(
        self, net: torch.nn.Module, metadata: Metadata, loss: Any = torch.nn.MSELoss()
    ):
        self.model_dir = metadata.model_dir
        self.model_path = os.path.join(self.model_dir, MODEL_FILENAME)

        self.net: torch.nn.Module = net
        self.loss = loss

    @abstractmethod
    def train_step(self, batch: Sequence[torch.Tensor]) -> float:
        pass

    @abstractmethod
    def compute_mse_loss(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        pass

    def load(self):
        self.net.load_state_dict(torch.load(self.model_path))

    def save(self):
        torch.save(self.net.state_dict(), self.model_path)

    def prepare_for_gpu(self):
        if torch.cuda.is_available():
            self.net.cuda()

    def num_parameters(self) -> int:
        if hasattr(self.net, "generator"):
            return sum(
                p.numel() for p in self.net.generator.parameters() if p.requires_grad
            )
        else:
            return sum(p.numel() for p in self.net.parameters() if p.requires_grad)
