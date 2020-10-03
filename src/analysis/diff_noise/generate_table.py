import json
from argparse import ArgumentParser, Namespace
from glob import glob
from pathlib import Path
from typing import List

import pandas as pd

ALIASES = {
    "none": "Brak",
    "mix": "Wszystkie",
    "zero_001": "Zero (0.01)",
    "zero_002": "Zero (0.02)",
    "zero": "Zero (0.05)",
    "gaussian_uni": "Gauss. pełny",
    "gaussian_part": "Gauss. cz.",
    "white_uni": "Równ. pełny",
    "white_part": "Równ. cz.",
}
MODELS = {"autoencoder": "Autoenkoder", "wavenet": "WaveNet", "segan": "SEGAN"}


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--pattern",
        type=str,
        default="models/**/small_model_*/metadata.json",
        help="Pattern for glob to find data",
    )
    parser.add_argument(
        "-l", "--loss_type", type=str, required=True, help="val or test"
    )
    parser.add_argument(
        "-o", "--output_path", type=str, required=True, help="Path to output plot"
    )

    return parser.parse_args()


def main(pattern: str, loss_type: str, output_path: str):
    model_paths = [Path(path) for path in glob(pattern)]

    df = get_df(model_paths, loss_type=loss_type)
    df = df.set_index("Model", drop=True).astype(float)
    df.to_csv(output_path, float_format="%.4f", index=False)


def get_df(model_paths: List[Path], loss_type: str) -> pd.DataFrame:
    rows = []
    for path in model_paths:
        with path.open("r") as f:
            metadata = json.load(f)
            noise_type = metadata["transformations"]
            if loss_type == "val":
                loss = metadata["final_val_loss"]
            elif loss_type == "test":
                loss = metadata["test_mse_loss"]
            else:
                raise ValueError(f"Unknown type: {loss_type}")
            row = {
                "Model": MODELS[metadata["model_dir"].split("/")[1]],
                "Szum": ALIASES[noise_type],
                "Błąd": loss,
                "Ziarno": metadata["random_seed"],
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    df = df.groupby(["Model", "Szum"], as_index=False).mean().drop("Ziarno", axis=1)
    separate = [df[df["Model"].str.contains(name)] for name in df["Model"].unique()]
    transformed = []
    for d in separate:
        name = d.iloc[0, 0]
        d.index = d["Szum"]
        d = d["Błąd"]
        d["Model"] = name
        transformed.append(d)
    df = pd.concat(transformed, axis=1).transpose().reset_index(drop=True)
    return df[["Model"] + list(ALIASES.values())]


if __name__ == "__main__":
    args = parse_args()
    main(pattern=args.pattern, output_path=args.output_path, loss_type=args.loss_type)
