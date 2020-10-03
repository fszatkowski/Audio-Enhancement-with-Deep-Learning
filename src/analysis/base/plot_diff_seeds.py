from argparse import ArgumentParser, Namespace
from glob import glob
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis.base.generate_table import get_df
from analysis.base.plot import create_single_plot
from analysis.plot import plot

COLOR_MAP = {
    "val": "lightblue",
    "test": "peachpuff",
    "base_val": "deepskyblue",
    "base_test": "orange",
    "mean_val": "blue",
    "mean_test": "red",
}

sns.set_style("darkgrid")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--pattern",
        type=str,
        default="models/autoencoder/base_model_*/metadata.json",
        help="Pattern for glob to find data",
    )
    parser.add_argument(
        "-o", "--output_path", type=str, required=True, help="Path to output plot"
    )
    parser.add_argument(
        "--title",
        "-t",
        default="Final loss against number of used files",
        type=str,
        help="Plot title",
    )
    parser.add_argument(
        "--logx",
        default=False,
        action="store_true",
        help="Use logarithmic scale for x axis",
    )
    parser.add_argument(
        "--logy",
        default=True,
        action="store_false",
        help="Use logarithmic scale for y axis",
    )

    return parser.parse_args()


def make_plot(
    pattern: str, output_path: str, title: str, log_x: bool = False, log_y: bool = True
):
    model_paths = glob(pattern)
    model_paths = [Path(path) for path in model_paths]

    df = get_df(model_paths, seeds=True)
    df["Błąd wal."] = 10 ** 4 * df["Błąd wal."]
    df["Błąd test."] = 10 ** 4 * df["Błąd test."]

    plot(
        output_path=output_path,
        function=create_plot,
        xlabel="Liczba plików",
        ylabel="Błąd [10e-4]",
        title="Błąd średniokwadratowy w zależności od liczby plików",
        logx=log_x,
        logy=log_y,
        xticks=np.array([256, 512, 1024, 2048, 4096, 8192], dtype=int),
        yticks=np.array([3, 4, 5, 6, 7]),
        df=df,
    )


def create_plot(df: pd.DataFrame) -> plt.Axes:
    base = df[df["Ziarno"] == 123]
    create_single_plot(
        base, "Autoenkoder", "val", "Bazowy [wal.]", COLOR_MAP["base_val"]
    )
    create_single_plot(
        base, "Autoenkoder", "test", "Bazowy [test.]", COLOR_MAP["base_test"]
    )

    mean = df.groupby(["Model", "Liczba plików uczących"], as_index=False).mean()
    mean["Model"] = "Autoenkoder"
    create_single_plot(
        mean, "Autoenkoder", "val", "Średnia [wal.]", COLOR_MAP["mean_val"]
    )
    create_single_plot(
        mean, "Autoenkoder", "test", "Średnia [test.]", COLOR_MAP["mean_test"]
    )

    rest = df[df["Ziarno"] != 123]
    create_single_plot(
        rest, "Autoenkoder", "val", "Pojedynczy [wal.]", COLOR_MAP["val"]
    )
    return create_single_plot(
        rest, "Autoenkoder", "test", "Pojedynczy [test.]", COLOR_MAP["test"]
    )


if __name__ == "__main__":
    args = parse_args()
    make_plot(
        pattern=args.pattern,
        output_path=args.output_path,
        title=args.title,
        log_x=args.logx,
        log_y=args.logy,
    )
