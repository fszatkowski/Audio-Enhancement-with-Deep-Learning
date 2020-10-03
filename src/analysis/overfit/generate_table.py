import json
from argparse import ArgumentParser, Namespace
from glob import glob
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

MODEL_NAME = {"autoencoder": "Autoenkoder", "segan": "SEGAN", "wavenet": "WaveNet"}


def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="results/overfit_models.csv",
        help="Path to output plot",
    )

    return parser.parse_args()


def main(output_path: str):
    model_paths = [
        Path(path) for path in glob("models/**/overfit_model*/metadata.json")
    ]
    df = get_df(model_paths)
    df.to_csv(output_path, float_format="%.4f", index=False)


def get_df(model_paths: List[Path]) -> pd.DataFrame:
    rows = []
    for path in model_paths:
        with path.open("r") as f:
            data = json.load(f)
            row = {
                "Model": MODEL_NAME[str(path).split("/")[1]],
                "L1": None,
                "Błąd": np.power(10, data["final_mse_loss"]),
                "Błąd [log10]": data["final_mse_loss"],
                "Liczba parametrów": data["num_params"],
                "Czas uczenia [h]": data["training_hours"],
                "Czas uczenia [min/epoka]": (
                    data["training_hours"] * 60 / data["current_epoch"]
                ),
            }
            if row["Model"] == "SEGAN":
                row["L1"] = str(path).split("/")[2].split("_")[-1]
            else:
                row["L1"] = "-"
            rows.append(row)

    return pd.DataFrame(rows).sort_values(by=["Model", "L1"], ascending=True)


if __name__ == "__main__":
    args = parse_args()
    main(output_path=args.output_path)
