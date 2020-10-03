import os
import warnings
from glob import glob
from typing import Optional, Tuple

import librosa
import numpy as np
import torch

from constants import MP3_GLOB

warnings.filterwarnings("ignore")


def get_mp3_files():
    return glob(MP3_GLOB)


def load_file(
    path: str, sample_rate: Optional[int] = None, mono: bool = False
) -> Tuple[torch.Tensor, int]:
    # sample rate = None means original sr is kept,
    # mono = False means original channels are kept
    try:
        audio, sr = librosa.load(path, sr=None, mono=mono)
        if audio.shape[1] % 2:
            audio = np.pad(audio, ((0, 0), (0, 1)))
        if sr != sample_rate and sample_rate is not None:
            audio = librosa.core.resample(
                audio, orig_sr=sr, target_sr=sample_rate, fix=True
            )
    except Exception as e:
        print(f"Error while loading from path {path}.")
        print(e)
        raise e

    # resampled audio sometimes has values outside <-1.0, 1.0> so they need to be fixed
    if audio.max() > 1.0:
        audio = np.minimum(1.0, audio)
    if audio.min() < -1.0:
        audio = np.maximum(-1.0, audio)
    audio = torch.from_numpy(audio)

    return audio, sr


def trim(audio: torch.Tensor):
    """ Remove silence from begining and from the end of file """
    indices = (audio[0, :] + audio[1, :]).nonzero()
    min_idx = indices.min()
    max_idx = indices.max()
    return audio[:, min_idx:max_idx]
