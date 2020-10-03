from common.dataloader import DataLoader
from common.dataset import Dataset
from common.metadata import Metadata


def assert_dataloader_correct(dataloader: DataLoader):
    shapes = None
    for dataloader_batch in dataloader:
        for batch in dataloader_batch:
            if shapes is None:
                shapes = [set() for _ in batch]
            for minibatch, shape in zip(batch, shapes):
                shape.add(minibatch.shape[-1])

    for shape in shapes:
        assert len(shape) == 1


def test_all_dataloaders():
    # test dataloaders for autoencoder, wavenet and segan
    # assert if they output correctly padded data (meaning we all outputs have the same dimension)
    dataset = Dataset(metadata=Metadata.get_mock())

    # autoencoder
    assert_dataloader_correct(DataLoader(Metadata.get_mock(), dataset, train_gan=False))

    # wavenet
    assert_dataloader_correct(DataLoader(Metadata.get_mock(), dataset, train_gan=False))

    # segan
    assert_dataloader_correct(DataLoader(Metadata.get_mock(), dataset, train_gan=True))


if __name__ == "__main__":
    test_all_dataloaders()
