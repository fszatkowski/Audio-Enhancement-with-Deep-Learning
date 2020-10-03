import json
import os
from argparse import ArgumentParser, Namespace
from shutil import copy

import librosa
import numpy as np
import torch

from common.transformations_manager import TransformationsManager
from data.utils import load_file
from eval.utils import process_nr
from inference.audio_processor import AudioProcessor


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--metadata",
        type=str,
        required=True,
        help="Path to trained model metadata",
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        required=True,
        help="Path to audio file to upsample",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True, help="Path to save output files"
    )
    parser.add_argument(
        "-n",
        "--noise",
        type=str,
        default=None,
        help="Optional transformation to use on file. "
        "The name should correspond to the ones used during training "
        "(eg. TranformationsManager.get_transformations() keys).",
    )
    parser.add_argument(
        "-s",
        "--sr",
        type=int,
        default=22050,
        help="If provided, "
        "will load file with this sampling rate "
        "instead of keeping original sampling rate.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.metadata, "r") as f:
        metadata = json.load(f)
    processor = AudioProcessor(metadata)

    os.makedirs(os.path.join(args.output_dir, args.noise), exist_ok=True)
    copy_path = os.path.join(args.output_dir, os.path.basename(args.input_file))
    copy(args.input_file, copy_path)

    noisy_file_out_path = os.path.join(args.output_dir, args.noise, "noisy_file.wav")
    processor.process_file(
        args.input_file,
        input_file_sr=args.sr,
        output_file_path=os.path.join(
            args.output_dir, args.noise, "nn_processed_file.wav"
        ),
        noise=args.noise,
        noisy_file_save_path=noisy_file_out_path,
    )

    noisy_audio, _ = load_file(noisy_file_out_path, 44100)
    transformation = TransformationsManager.get_transformations(args.noise)[0]
    transformation.apply_probability = 1.0
    noise_estimation = transformation.apply(torch.zeros(noisy_audio.shape))

    nr_denoised_estimated_noise = process_nr(noisy_audio, noise_estimation)
    librosa.output.write_wav(
        os.path.join(args.output_dir, args.noise, "benchmark_noise_file.wav"),
        y=np.asfortranarray(nr_denoised_estimated_noise),
        sr=44100,
    )

    nr_denoised_no_noise = process_nr(noisy_audio, torch.zeros(noisy_audio.shape))
    librosa.output.write_wav(
        os.path.join(args.output_dir, args.noise, "benchmark_no_noise_file.wav"),
        y=np.asfortranarray(nr_denoised_no_noise),
        sr=44100,
    )
