import torch

from autoencoder.metadata import AutoencoderMetadata
from autoencoder.model_wrapper import AutoencoderWrapper
from common.training import run_full_pipeline
from common.utils import ModelType


def train(metadata: AutoencoderMetadata):
    autoencoder = AutoencoderWrapper(metadata, loss=torch.nn.MSELoss())

    run_full_pipeline(
        metadata=metadata, model_wrapper=autoencoder, model_type=ModelType.Autoencoder
    )


if __name__ == "__main__":
    train(AutoencoderMetadata.from_args())
