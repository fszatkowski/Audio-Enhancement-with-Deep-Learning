import glob
import json

import numpy as np
import pandas as pd
import typer

from analysis.diff_noise.generate_table import ALIASES

cli = typer.Typer()


@cli.command()
def generate_table(
    pattern: str = "models/autoencoder/final*/metadata.json",
    output: str = "results/final/results.csv",
    average: bool = False,
):
    df = get_df(pattern, average)
    df.to_csv(output, float_format="%.6f", index=False)


def get_df(input_pattern: str, average_seeds: bool = True):
    paths = glob.glob(input_pattern)
    rows = []
    for path in paths:
        with open(path, "r") as f:
            metadata = json.load(f)
            try:
                row = {
                    "Błąd wal.": np.power(10, metadata["final_val_loss"]),
                    "Błąd test.": np.power(10, metadata["test_mse_loss"]),
                    "Błąd wal.[log10]": metadata["final_val_loss"],
                    "Błąd test.[log10]": metadata["test_mse_loss"],
                    "Rodzaj szumu": ALIASES[metadata["transformations"]],
                    "Ziarno": metadata["random_seed"],
                }
                rows.append(row)
            except KeyError:
                continue
    df = pd.DataFrame(rows)
    if average_seeds:
        df = df.drop(["Ziarno"], axis=1)
        df = df.groupby(["Rodzaj szumu"]).mean().reset_index()
        df["Błąd wal.[log10]"] = np.log10(df["Błąd wal."])
        df["Błąd test.[log10]"] = np.log10(df["Błąd test."])
    df = df.sort_values(by="Błąd test.[log10]", ascending=True)
    return df


if __name__ == "__main__":
    typer.run(generate_table)
