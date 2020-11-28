from typing import Dict, Generator, List, Optional, Sequence, Tuple

import torch
import torch.utils.data as data

from common.dataset import Dataset
from common.metadata import Metadata
from common.transformations_manager import TransformationsManager
from constants import NUM_WORKERS


class DataLoader(data.DataLoader):
    def __init__(
        self,
        metadata: Metadata,
        dataset: Dataset,
        num_workers: int = NUM_WORKERS,
        train_gan: bool = False,
    ):
        super(DataLoader, self).__init__(
            dataset, metadata.batch_size, shuffle=True, num_workers=num_workers
        )

        self.metadata = metadata

        self.input_samples = metadata.input_samples
        self.input_step = metadata.input_step_samples

        self.target_samples = metadata.target_samples
        self.target_step = metadata.target_step_samples

        self.collate_fn = self._collate_fn
        self.train_gan = train_gan
        self.tm: Optional[TransformationsManager] = None
        self.tm_active = False

    def set_transformations_manager(self, tm: TransformationsManager):
        self.tm = tm
        self.tm_active = True

    def _collate_fn(
        self, raw_data: Sequence[Dict[str, List[torch.Tensor]]]
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        x, y = self._prepare(raw_data)

        while x.shape[3] and y.shape[3]:
            inputs = x[:, :, :, : self.input_samples]
            targets = y[:, :, :, : self.target_samples]

            if self.train_gan:
                distribution_targets = shuffle_batch(targets)

                if torch.cuda.is_available():
                    yield inputs.cuda(), targets.cuda(), distribution_targets.cuda()
                else:
                    yield inputs, targets, distribution_targets
            else:
                if self.tm is not None and self.tm_active:
                    inputs = self.tm.apply_transformations(inputs)

                if torch.cuda.is_available():
                    yield inputs.cuda(), targets.cuda()
                else:
                    yield inputs, targets

            x = x[:, :, :, self.input_step :]
            y = y[:, :, :, self.target_step :]

            # stop outputting data when we reach the end
            if x.shape[-1] < self.input_samples:
                break

    def _prepare(
        self, dataset_output: Sequence[Dict[str, List[torch.Tensor]]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_list = [torch.stack(d["x"], 0) for d in dataset_output]
        y_list = [torch.stack(d["y"], 0) for d in dataset_output]

        max_x_shape = 0
        max_y_shape = 0
        for x, y in zip(x_list, y_list):
            if x.shape[-1] > max_x_shape:
                max_x_shape = x.shape[-1]
                max_y_shape = y.shape[-1]

        x_shape_to_pad = self.input_step * (
            max_x_shape // self.input_step
            + (1 if max_x_shape % self.input_step != 0 else 0)
        )
        y_shape_to_pad = self.target_step * (
            max_y_shape // self.target_step
            + (1 if max_x_shape % self.target_step != 0 else 0)
        )
        for i, (x, y) in enumerate(zip(x_list, y_list)):
            x_pad = (0, x_shape_to_pad - x.shape[-1])
            y_pad = (0, y_shape_to_pad - y.shape[-1])

            x_list[i] = torch.nn.functional.pad(x, x_pad, mode="constant", value=0)
            y_list[i] = torch.nn.functional.pad(y, y_pad, mode="constant", value=0)

        return torch.stack(x_list, dim=0), torch.stack(y_list, dim=0)


def shuffle_batch(tensor: torch.Tensor) -> torch.Tensor:
    shuffled = torch.zeros(tensor.shape)
    for i, batch_idx in enumerate(torch.randperm(tensor.shape[0])):
        shuffled[i, :, :, :] = tensor[batch_idx, :, :, :]
    return shuffled
