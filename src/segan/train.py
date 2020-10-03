import torch

from common.training import run_full_pipeline
from common.utils import ModelType
from segan.metadata import SEGANMetadata
from segan.model_wrapper import SEGANWrapper


def train(metadata: SEGANMetadata):
    segan = SEGANWrapper(metadata, loss=torch.nn.MSELoss())

    run_full_pipeline(
        metadata=metadata, model_wrapper=segan, model_type=ModelType.SEGAN
    )


if __name__ == "__main__":
    train(SEGANMetadata.from_args())
