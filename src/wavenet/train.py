import torch

from common.training import run_full_pipeline
from common.utils import ModelType
from wavenet.metadata import WaveNetMetadata
from wavenet.model_wrapper import WaveNetWrapper


def train(metadata: WaveNetMetadata):
    wavenet = WaveNetWrapper(metadata, loss=torch.nn.MSELoss())

    run_full_pipeline(
        metadata=metadata, model_wrapper=wavenet, model_type=ModelType.WaveNet
    )


if __name__ == "__main__":
    train(WaveNetMetadata.from_args())
