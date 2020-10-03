import random
from glob import glob
from typing import Dict, Sequence, Tuple, Union

import torch
import torch.utils.data as data

from common.metadata import Metadata
from constants import PREPROCESSED_DATASET_DIR, RAW_DATASET_FILELIST
from data.utils import load_file


class Dataset(data.Dataset):
    def __init__(self, metadata: Metadata):
        super(Dataset, self).__init__()

        self.x_sr = metadata.input_sr
        self.y_sr = metadata.target_sr

        random.seed(metadata.random_seed)

        # if preprocessed dataset is present, load pt files
        self.load_raw = False
        self.files = glob(PREPROCESSED_DATASET_DIR + "/*.pt")

        # if there are no pt files, load and preprocess mp3s
        if len(self.files) == 0:
            with open(RAW_DATASET_FILELIST, "r") as f:
                self.files = [file.strip() for file in f.readlines()]
                self.load_raw = True
        else:
            self.files.sort()

        random.shuffle(self.files)

        if metadata.train_files != 0:
            if (metadata.test_files + metadata.val_files + metadata.train_files) > len(
                self.files
            ):
                raise ValueError(
                    f"Trying to create dataset from {metadata.test_files + metadata.val_files + metadata.train_files} files while only {len(self.files)} files are available."
                )
            self.files = self.files[
                : metadata.test_files + metadata.val_files + metadata.train_files
            ]
        self.length, self.batch_size = self._get_sizes()

        print(f"Using {self.length} examples stored in packs of {self.batch_size}.")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: Union[int, slice]) -> Dict[str, Sequence[torch.Tensor]]:
        if isinstance(idx, int):
            indices = [idx]
        elif isinstance(idx, slice):
            indices = list(range(*idx.indices(self.length)))
        else:
            raise ValueError("Dataset should be indexed with either slice or int.")
        x_list = []
        y_list = []
        for index in indices:
            x, y = self._get_one_pair(index)
            x_list.append(x)
            y_list.append(y)
        return {"x": x_list, "y": y_list}

    def _get_sizes(self) -> Tuple[int, int]:
        if not self.load_raw:
            batch_size = len(torch.load(self.files[0]))
            last_batch_size = len(torch.load(self.files[-1]))
            return batch_size * (len(self.files) - 1) + last_batch_size, batch_size
        else:
            return len(self.files), 1

    def _get_one_pair(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.load_raw:
            batch_file_idx = idx // self.batch_size
            audio_idx = idx % self.batch_size
            x, y = torch.load(self.files[batch_file_idx])[audio_idx]
        else:
            y, _ = load_file(self.files[idx], 44100)
            x, _ = load_file(self.files[idx], 22050)
        return x, y
