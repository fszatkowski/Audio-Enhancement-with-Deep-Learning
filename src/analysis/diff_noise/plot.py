import json
from argparse import ArgumentParser, Namespace
from glob import glob
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import seaborn as sns

from analysis.plot import plot

sns.set_style("darkgrid")
COLOR = {"val": "Blues", "test": "Greens"}
ALIASES = {
    "none": "Brak",
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
        default="results/**/small_model*/metadata.json",
        help="Pattern to seek models",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="results/diff_noise_results",
        help="Path to output plot",
    )
    parser.add_argument(
        "--abs_loss",
        default=True,
        action="store_false",
        help="Use absolute values for loss instead of using logarithm",
    )
    parser.add_argument("-l", "--loss_type", default="val", help="Either val or test")
    return parser.parse_args()


def make_plot(pattern: str, output_path: str, loss_type: str, log_loss: bool):
    model_paths = [Path(path) for path in glob(pattern)]
    df = create_df(model_paths)

    df1 = df[df["noise"].isin(["Brak", "Zero (0.01)", "Zero (0.02)", "Zero (0.05)"])]
    df2 = df[~df["noise"].isin(["Brak", "Zero (0.01)", "Zero (0.02)", "Zero (0.05)"])]

    plot(
        f"{output_path}_1",
        sns.barplot,
        title="Błąd w zależności od rodzaju zniekształceń",
        xlabel="Rodzaj zniekształcenia",
        ylabel="Błąd[10e-3]",
        logy=log_loss,
        yticks=np.array([1, 3, 10, 30, 100, 300]),
        x="noise",
        y=f"{loss_type}_loss",
        hue="model",
        palette=COLOR[loss_type],
        data=df1,
        xtext=True,
    )

    plot(
        f"{output_path}_2",
        sns.barplot,
        title="Błąd w zależności od rodzaju zniekształceń",
        xlabel="Rodzaj zniekształcenia",
        ylabel="Błąd[10e-3]",
        logy=log_loss,
        yticks=np.array([1, 3, 10, 30, 100, 300]),
        x="noise",
        y=f"{loss_type}_loss",
        hue="model",
        palette=COLOR[loss_type],
        data=df2,
        xtext=True,
    )


def create_df(paths: Iterator[Path]) -> pd.DataFrame:
    rows = []

    for path in paths:
        with path.open("r") as f:
            metadata = json.load(f)
            transformation = metadata["transformations"]
            row = {
                "model": MODELS[str(path).split("/")[1]],
                "noise": ALIASES[transformation],
                "val_loss": 10e3 * np.power(10, metadata["final_val_loss"]),
                "test_loss": 10e3 * np.power(10, metadata["test_mse_loss"]),
            }

            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    args = parse_args()
    make_plot(
        pattern=args.pattern,
        output_path=args.output_path,
        loss_type=args.loss_type,
        log_loss=args.abs_loss,
    )
