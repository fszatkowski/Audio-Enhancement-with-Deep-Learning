import json
from argparse import ArgumentParser, Namespace
from glob import glob
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis.base.generate_table import get_df
from analysis.plot import plot
from constants import COLOR_MAP


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--pattern",
        type=str,
        default="models/**/base_model_*/metadata.json",
        help="Pattern for glob to find data",
    )
    parser.add_argument(
        "-o", "--output_path", type=str, required=True, help="Path to output plot"
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


def make_plot(pattern: str, output_path: str, log_x: bool = False, log_y: bool = True):
    model_paths = glob(pattern)
    model_paths = [Path(path) for path in model_paths if "seed" not in str(path)]

    df = get_df(model_paths)
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
        yticks=np.array([2, 3, 4, 5, 6]),
        df=df,
    )


def create_plot(df: pd.DataFrame) -> plt.Axes:
    create_single_plot(
        df, "Autoenkoder", "val", "Autoenkoder[wal.]", COLOR_MAP["autoencoder_val"]
    )
    create_single_plot(
        df, "Autoenkoder", "test", "Autoenkoder[test.]", COLOR_MAP["autoencoder_test"]
    )

    create_single_plot(df, "WaveNet", "val", "WaveNet[wal.]", COLOR_MAP["wavenet_val"])
    create_single_plot(
        df, "WaveNet", "test", "WaveNet[test.]", COLOR_MAP["wavenet_test"]
    )

    create_single_plot(df, "SEGAN", "val", "SEGAN[wal.]", COLOR_MAP["segan_val"])
    return create_single_plot(
        df, "SEGAN", "test", "SEGAN[test.]", COLOR_MAP["segan_test"]
    )


def create_single_plot(
    df: pd.DataFrame, model: str, loss_type: str, label: str, color: str
):
    df = df[df["Model"].str.contains(model)]
    if loss_type == "val":
        ycol = "Błąd wal."
    elif loss_type == "test":
        ycol = "Błąd test."
    else:
        raise ValueError()

    return sns.lineplot(
        x="Liczba plików uczących", y=ycol, data=df, label=label, color=color
    )


def get_data(
    paths: Iterator[Path], log_x: bool = False, log_y: bool = False
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    val_losses_data: Dict[float, List[float, float]] = {}
    test_losses_data: Dict[float, List[float, float]] = {}

    for path in paths:
        with path.open("r") as f:
            metadata = json.load(f)
            train_files = metadata["train_files"]
            if train_files == 0:
                train_files = 7490 - metadata["test_files"] - metadata["val_files"]
            try:
                val_loss = metadata["final_val_loss"]
                test_loss = metadata["test_mse_loss"]
            except KeyError:
                continue

            if log_x:
                train_files = np.log2(train_files)

            val_loss = np.power(10, val_loss)
            test_loss = np.power(10, test_loss)

            if train_files not in val_losses_data:
                val_losses_data[train_files] = [val_loss, 1]
            else:
                val_losses_data[train_files][0] += val_loss
                val_losses_data[train_files][1] += 1

            if train_files not in test_losses_data:
                test_losses_data[train_files] = [test_loss, 1]
            else:
                test_losses_data[train_files][0] += test_loss
                test_losses_data[train_files][1] += 1

    for key, value in val_losses_data.items():
        val_losses_data[key] = [value[0] / value[1], 1]
    for key, value in test_losses_data.items():
        test_losses_data[key] = [value[0] / value[1], 1]

    val_losses_data: List[Tuple[float, float]] = [
        (key, value[0]) for key, value in val_losses_data.items()
    ]
    test_losses_data: List[Tuple[float, float]] = [
        (key, value[0]) for key, value in test_losses_data.items()
    ]

    val_losses_data.sort(key=lambda el: el[0])
    test_losses_data.sort(key=lambda el: el[0])

    if log_y:
        val_losses_data = list(map(lambda e: (e[0], np.log10(e[1])), val_losses_data))
        test_losses_data = list(map(lambda e: (e[0], np.log10(e[1])), test_losses_data))

    return val_losses_data, test_losses_data


if __name__ == "__main__":
    args = parse_args()
    make_plot(
        pattern=args.pattern,
        output_path=args.output_path,
        log_x=args.logx,
        log_y=args.logy,
    )
