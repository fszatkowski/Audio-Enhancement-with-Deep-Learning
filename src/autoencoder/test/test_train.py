import shutil

from autoencoder.metadata import AutoencoderMetadata
from autoencoder.train import train


def test_train():
    test_dir = "models/autoencoder/unit_test"
    metadata = AutoencoderMetadata(
        epochs=4,
        warmup_epochs=2,
        patience=4,
        transformations="default",
        batch_size=1,
        train_files=3,
        test_files=2,
        val_files=1,
        save_every_n_steps=100,
        max_transformations_applied=2,
        model_dir=test_dir,
        num_layers=3,
        channels=2,
        kernel_size=3,
        multiplier=2,
        activation="prelu",
        norm="batch_norm",
    )
    train(metadata)
    shutil.rmtree(test_dir)


if __name__ == "__main__":
    test_train()
