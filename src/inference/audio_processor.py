from math import ceil
from typing import Any, Dict, Optional

import librosa
import numpy as np
import torch
from tqdm import tqdm

from common.transformations_manager import TransformationsManager
from inference.restore_model import restore_model
from wavenet.modules import WaveNetModule


class AudioProcessor:
    def __init__(self, metadata: Dict[str, Any], batch_size: int = 16):

        torch.set_grad_enabled(False)
        self.model = restore_model(metadata=metadata)
        self.batch_size = batch_size

        self.input_sr = metadata["input_sr"]
        self.target_sr = metadata["target_sr"]
        self.input_samples = metadata["input_samples"]
        self.target_samples = metadata["target_samples"]
        self.input_step = metadata["input_step_samples"]
        self.target_step = metadata["target_step_samples"]

        if isinstance(self.model, WaveNetModule):
            self.target_receptive_fields = 2 * self.model.receptive_fields
        else:
            self.target_receptive_fields = 0

    def process_file(
        self,
        input_file_path: str,
        input_file_sr: int = 0,
        weights: Optional[np.array] = None,
        output_file_path: Optional[str] = None,
        return_array: Optional[bool] = False,
        noise: Optional[str] = None,
        noisy_file_save_path: Optional[str] = None,
    ):
        if input_file_sr:
            audio, _ = librosa.load(input_file_path, sr=input_file_sr, mono=False)
        else:
            audio, sr = librosa.load(input_file_path, sr=None, mono=False)
            if sr != self.input_sr:
                raise ValueError(
                    f"File with sampling rate {sr} while {self.input_sr} is expected."
                )

        if noise:
            transformations = TransformationsManager.get_transformations(noise)
            if len(transformations) > 1:
                raise ValueError(f"Transformation: {noise} refers to a sequence.")
            transformation = transformations[0]
            transformation.apply_probability = 1.0
            audio = transformation.apply(torch.Tensor(audio))

        if noisy_file_save_path is not None:
            librosa.output.write_wav(
                noisy_file_save_path, y=np.asfortranarray(audio), sr=input_file_sr
            )

        output_array = self.process_array(audio, weights)
        if output_file_path is not None:
            librosa.output.write_wav(
                output_file_path,
                y=np.asfortranarray(np.squeeze(output_array)),
                sr=self.target_sr,
            )

        if return_array:
            return output_array

    def process_array(
        self,
        input_array: np.array,
        weights: Optional[np.array] = None,
        verbose: bool = False,
    ) -> np.array:
        input_array = AudioProcessor.correct_shape(input_array)
        x_pad = (
            self.input_step * ceil(input_array.shape[-1] / self.input_step)
            - input_array.shape[-1]
        )
        input_array = np.pad(
            input_array,
            ((0, 0), (0, 0), (0, 0), (0, x_pad)),
            mode="constant",
            constant_values=0,
        )
        output_array = np.zeros((*input_array.shape[:-1], 2 * input_array.shape[-1]))

        weights_sum = np.zeros(output_array.shape)
        if weights is None:
            weights = np.ones(
                (
                    *input_array.shape[:-1],
                    self.target_samples - self.target_receptive_fields,
                )
            )
            weights_sum = weights_sum + 1

        current_x = 0
        current_y = 0

        steps = range(self.calculate_total_steps(input_array.shape[-1]))
        if verbose:
            steps = tqdm(steps, desc="Processing file...")
        for _ in steps:
            batch = []
            for i in range(self.batch_size):
                sample = input_array[
                    :,
                    :,
                    :,
                    current_x
                    + i * self.input_step : current_x
                    + i * self.input_step
                    + self.input_samples,
                ]
                if sample.shape[-1] == self.input_samples:
                    batch.append(sample)

            batch = torch.Tensor(np.concatenate(batch, axis=0))
            if torch.cuda.is_available():
                batch = batch.cuda()

            model_out = self.model(batch).cpu().numpy()

            for i in range(batch.shape[0]):
                output_array[
                    :,
                    :,
                    :,
                    current_y
                    + self.target_receptive_fields : current_y
                    + self.target_samples,
                ] += (weights * model_out[i])
                weights_sum[
                    :,
                    :,
                    :,
                    current_y
                    + self.target_receptive_fields : current_y
                    + self.target_samples,
                ] += weights
                current_x += self.input_step
                current_y += self.target_step

        scaled_output = output_array / weights_sum
        return np.squeeze(scaled_output[:, :, :, : -2 * x_pad])

    def calculate_total_steps(self, input_length: int) -> int:
        ctr = 0
        length = input_length
        while length >= self.input_samples:
            length -= self.input_step
            ctr += 1
        return ceil(ctr / self.batch_size)

    @staticmethod
    def correct_shape(input_array: np.array) -> np.array:
        if input_array.shape[-2] != 2:
            if input_array.shape[-2] == 1:
                print("WARNING: Converting mono signal to stereo.")
                input_array = np.repeat(input_array, repeats=2, axis=-2)
            else:
                raise ValueError(
                    f"Expected 2 audio channels, was provided: {input_array.shape[-2]} channels."
                )
        if len(input_array.shape) < 2:
            raise ValueError(
                f"Input shape: {input_array.shape}, while shape (1, 1, 2, *N*) is required."
            )
        elif len(input_array.shape) == 2:
            input_array = input_array[np.newaxis, np.newaxis, :, :]
        elif len(input_array.shape) == 3:
            input_array = input_array[np.newaxis, :, :, :]
        return input_array
