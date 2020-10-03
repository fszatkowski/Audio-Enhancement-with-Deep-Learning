import torch

import constants
from autoencoder.metadata import AutoencoderMetadata
from autoencoder.model_wrapper import AutoencoderWrapper
from common.training import run_overfit
from common.utils import ModelType


def train(metadata: AutoencoderMetadata):
    run_overfit(
        metadata=metadata,
        model_wrapper=AutoencoderWrapper(metadata, loss=torch.nn.MSELoss()),
        model_type=ModelType.Autoencoder,
    )


if __name__ == "__main__":
    train(
        AutoencoderMetadata(
            epochs=constants.OVERFIT_EPOCHS,
            patience=constants.OVERFIT_PATIENCE,
            batch_size=constants.OVERFIT_BATCH_SIZE,
            train_files=32,
            val_files=1,
            test_files=1,
            save_every_n_steps=constants.OVERFIT_N_STEPS,
            transformations="none",
            model_dir="models/autoencoder/overfit_model",
        )
    )
