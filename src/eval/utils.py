from dataclasses import dataclass
from typing import Dict, List

import noisereduce as nr
import numpy as np

from common.dataset import Dataset
from common.transformations import (GaussianNoisePartial, GaussianNoiseUniform,
                                    WhiteNoisePartial, WhiteNoiseUniform,
                                    ZeroSamplesTransformation)

TRANSFORMATIONS = {
    "none": None,
    "gaussian_part": GaussianNoisePartial(1, noise_percent=0.5, mean=0.0, std=0.1),
    "gaussian_uni": GaussianNoiseUniform(1, mean=0.0, std=0.1),
    "white_part": WhiteNoisePartial(1, noise_percent=0.5, amplitude=0.1),
    "white_uni": WhiteNoiseUniform(1, amplitude=0.1),
    "zero_001": ZeroSamplesTransformation(1, noise_percent=0.01),
    "zero_002": ZeroSamplesTransformation(1, noise_percent=0.02),
    "zero_005": ZeroSamplesTransformation(1, noise_percent=0.05),
}


@dataclass
class EvalResult:
    mse: float
    mae: float
    snr: float
    psnr: float

    @staticmethod
    def calculate(clean: np.array, noisy: np.array) -> "EvalResult":
        mse = float(((clean - noisy) ** 2).mean())
        mae = float(np.abs(clean - noisy).mean())
        snr = _snr(clean, noisy)
        psnr = _psnr(clean, noisy)
        return EvalResult(mse, mae, snr, psnr)


def _snr(clean: np.array, noisy: np.array) -> float:
    noise = clean - noisy
    p_sig = np.power(clean, 2).mean()
    p_n = np.power(noise, 2).mean()
    return float(10 * np.log10(p_sig / p_n))


def _psnr(clean: np.array, noisy: np.array) -> float:
    noise = clean - noisy
    mx = np.max(np.abs(clean))
    mse = np.power(noise, 2).mean()
    return float(10 * np.log10(np.power(mx, 2) / mse))


def get_mean_eval_results(results_list: List[EvalResult]) -> EvalResult:
    cumulative = EvalResult(0, 0, 0, 0)
    for results in results_list:
        cumulative.mse += results.mse
        cumulative.mae += results.mae
        cumulative.snr += results.snr
        cumulative.psnr += results.psnr

    cumulative.mse = cumulative.mse / len(results_list)
    cumulative.mae = cumulative.mae / len(results_list)
    cumulative.snr = cumulative.snr / len(results_list)
    cumulative.psnr = cumulative.psnr / len(results_list)

    return cumulative


def process_nr(input_clip: np.array, noise: np.array) -> np.array:
    processed = np.zeros(input_clip.shape)
    processed[0, :] = nr.reduce_noise(
        audio_clip=np.asfortranarray(input_clip[0, :]),
        noise_clip=np.asfortranarray(noise[0, :]),
        verbose=False,
    )
    processed[1, :] = nr.reduce_noise(
        audio_clip=np.asfortranarray(input_clip[1, :]),
        noise_clip=np.asfortranarray(noise[1, :]),
        verbose=False,
    )
    return processed


def get_test_set_files(metadata: Dict) -> List[str]:
    @dataclass
    class MockMetadata:
        input_sr: int
        target_sr: int
        random_seed: int
        train_files: int
        val_files: int
        test_files: int

    metadata = MockMetadata(
        input_sr=metadata["input_sr"],
        target_sr=metadata["target_sr"],
        random_seed=metadata["random_seed"],
        train_files=metadata["train_files"],
        val_files=metadata["val_files"],
        test_files=metadata["test_files"],
    )
    dataset = Dataset(metadata)
    test_files = dataset.files[: metadata.test_files]
    return test_files
