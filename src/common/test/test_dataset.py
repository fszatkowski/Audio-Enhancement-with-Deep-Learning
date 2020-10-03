from common.dataset import Dataset
from common.metadata import Metadata


def test_dataset():
    metadata = Metadata.get_mock()
    metadata.train_files = 8

    dataset = Dataset(metadata=metadata)
    sample_single_item = dataset[0]
    assert sample_single_item["x"][0].shape[0] == 2
    assert sample_single_item["y"][0].shape[0] == 2
    assert (
        sample_single_item["x"][0].shape[1] * 2 == sample_single_item["y"][0].shape[1]
    )

    sample_multiple_items = dataset[-8:-5]
    assert len(sample_multiple_items["x"]) == 3


if __name__ == "__main__":
    test_dataset()
