import json
from argparse import ArgumentParser, Namespace
from glob import glob
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

PRECISION_TEMPLATE = "%.4f"
MODEL_NAME = {"autoencoder": "Autoenkoder", "segan": "SEGAN", "wavenet": "WaveNet"}


def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="results/base_models.csv",
        help="Path to output plot",
    )

    return parser.parse_args()


def main(output_path: str):
    model_paths = [
        Path(path)
        for path in glob("models/**/base_model*/metadata.json")
        if "seed" not in path
    ]

    df = get_df(model_paths)
    df.sort_values(by=["Model", "Liczba plików uczących"], ascending=True)
    df.to_csv(output_path, float_format="%.4f", index=False)


def get_df(model_paths: List, seeds: bool = False) -> pd.DataFrame:
    rows = []
    for path in model_paths:
        with path.open("r") as f:
            try:
                data = json.load(f)
                row = {
                    "Model": MODEL_NAME[str(path).split("/")[1]],
                    "Liczba plików uczących": data["train_files"],
                    "Błąd wal.": np.power(10, data["final_val_loss"]),
                    "Błąd test.": np.power(10, data["test_mse_loss"]),
                    "Błąd wal. [log10]": data["final_val_loss"],
                    "Błąd test. [log10]": data["test_mse_loss"],
                    "Czas uczenia [h/epoka]": data["training_hours"]
                    / data["current_epoch"],
                }
                if row["Liczba plików uczących"] == 0:
                    row["Liczba plików uczących"] = 7490 - 512
                row["Czas uczenia [min/(epoka*plik)]"] = (
                    data["training_hours"]
                    * 60
                    / data["current_epoch"]
                    / row["Liczba plików uczących"]
                )
                if seeds:
                    row["Ziarno"] = data.get("random_seed", 123)

                rows.append(row)
            except KeyError:
                continue
    return pd.DataFrame(rows)


if __name__ == "__main__":
    args = parse_args()
    main(output_path=args.output_path)
