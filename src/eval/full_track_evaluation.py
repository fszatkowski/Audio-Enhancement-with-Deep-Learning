import json
from copy import deepcopy
from time import time

import librosa
import numpy as np
import pandas as pd
import torch
import typer
from tqdm import tqdm

from data.utils import load_file
from eval.utils import (TRANSFORMATIONS, EvalResult, get_mean_eval_results,
                        get_test_set_files, process_nr)
from inference.audio_processor import AudioProcessor
from inference.restore_weights import get_weights

cli = typer.Typer()


@cli.command()
def main(
    metadata: str = "models/autoencoder/final_997_none/metadata.json",
    output: str = "full_file_eval_test.csv",
    transformations: str = "none",
    weights: str = "tri",
    batch_size: int = 32,
):
    with open(metadata, "r") as f:
        metadata = json.load(f)
    processor = AudioProcessor(metadata, batch_size)
    weights_array = get_weights(weights, metadata["target_samples"])

    input_file_list = get_test_set_files(metadata)
    transformations = {t: TRANSFORMATIONS[t] for t in transformations.split(" ")}

    model_results = {
        transformation_name: [] for transformation_name in transformations.keys()
    }
    benchmark_results = deepcopy(model_results)
    no_denoising_results = deepcopy(model_results)

    for file in tqdm(input_file_list, desc="Processing files..."):
        if file.endswith(".pt"):
            original_audio = torch.load(file)[0][1].numpy()
            downsampled_audio = librosa.resample(original_audio, 44100, 22050)
        else:
            original_audio = load_file(file, 44100)
            downsampled_audio = load_file(file, 22050)

        for transformation_name, transformation in transformations.items():
            if transformation is not None:
                noisy_audio = transformation.apply(
                    torch.Tensor(downsampled_audio)
                ).numpy()
                model_out = processor.process_array(noisy_audio, weights_array)

                noise = noisy_audio - downsampled_audio
                benchmark = process_nr(noisy_audio, noise)
                benchmark = librosa.resample(np.asfortranarray(benchmark), 22050, 44100)

                no_denoising_audio = librosa.resample(noisy_audio, 22050, 44100)
                no_denoising_results[transformation_name].append(
                    EvalResult.calculate(original_audio, no_denoising_audio)
                )
            else:
                model_out = processor.process_array(downsampled_audio, weights_array)
                benchmark = librosa.resample(downsampled_audio, 22050, 44100)

            model_results[transformation_name].append(
                EvalResult.calculate(original_audio, model_out)
            )
            benchmark_results[transformation_name].append(
                EvalResult.calculate(original_audio, benchmark)
            )
    rows = []

    for transformation_name in transformations.keys():
        model_results_row = get_mean_eval_results(model_results[transformation_name])
        model_results_row.__dict__["name"] = f"{transformation_name}_model"
        benchmark_results_row = get_mean_eval_results(
            benchmark_results[transformation_name]
        )
        benchmark_results_row.__dict__["name"] = f"{transformation_name}_benchmark"

        rows.extend([model_results_row.__dict__, benchmark_results_row.__dict__])

        if transformation_name != "none":
            no_denoising_results_row = get_mean_eval_results(
                no_denoising_results[transformation_name]
            )
            no_denoising_results_row.__dict__[
                "name"
            ] = f"{transformation_name}_no_denoising"
            rows.append(no_denoising_results_row.__dict__)

    df = pd.DataFrame(rows)
    df.to_csv(output, float_format="%.8f", index=False)


if __name__ == "__main__":
    typer.run(main)
