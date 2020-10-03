from enum import Enum
from typing import Sequence, Tuple

import torch

from common.dataloader import DataLoader
from common.dataset import Dataset
from common.metadata import Metadata
from common.transformations import Transformation
from common.transformations_manager import TransformationsManager


class ModelType(Enum):
    Autoencoder = 0
    WaveNet = 1
    SEGAN = 2


def create_data_loaders(
    dataset: Dataset,
    metadata: Metadata,
    model: ModelType,
    transformations: Sequence[Transformation] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    test_set = torch.utils.data.Subset(dataset, range(metadata.test_files))
    val_set = torch.utils.data.Subset(
        dataset, range(metadata.test_files, metadata.test_files + metadata.val_files)
    )
    train_set = torch.utils.data.Subset(
        dataset, range(metadata.test_files + metadata.val_files, len(dataset))
    )

    if model == ModelType.Autoencoder or model == ModelType.WaveNet:
        loaders = (
            DataLoader(metadata, train_set),
            DataLoader(metadata, val_set),
            DataLoader(metadata, test_set),
        )

    elif model == ModelType.SEGAN:
        loaders = (
            DataLoader(metadata, train_set, train_gan=True),
            DataLoader(metadata, val_set),
            DataLoader(metadata, test_set),
        )
    else:
        raise ValueError("Model type not supported.")

    if transformations is not None:
        for loader in loaders:
            loader.set_transformations_manager(TransformationsManager(transformations))

    return loaders


def l1_regularization(model: torch.nn.Module) -> torch.Tensor:
    l1 = torch.tensor(0.0, requires_grad=True)
    for name, param in model.named_parameters():
        if "weight" in name:
            l1 += torch.norm(param, 1)
    return l1
