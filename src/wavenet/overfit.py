import torch

import constants
from common.training import run_overfit
from common.utils import ModelType
from wavenet.metadata import WaveNetMetadata
from wavenet.model_wrapper import WaveNetWrapper


def train(metadata: WaveNetMetadata):
    run_overfit(
        metadata=metadata,
        model_wrapper=WaveNetWrapper(metadata, loss=torch.nn.MSELoss()),
        model_type=ModelType.WaveNet,
    )


if __name__ == "__main__":
    train(
        WaveNetMetadata(
            epochs=constants.OVERFIT_EPOCHS,
            patience=constants.OVERFIT_PATIENCE,
            batch_size=constants.OVERFIT_BATCH_SIZE,
            train_files=32,
            val_files=1,
            test_files=1,
            save_every_n_steps=constants.OVERFIT_N_STEPS,
            transformations="none",
            model_dir="models/wavenet/overfit_model",
        )
    )
