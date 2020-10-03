import glob
import json

import numpy as np
import pandas as pd
import typer

cli = typer.Typer()

MODEL = {"autoencoder": "Autoenkoder", "wavenet": "WaveNet", "segan": "SEGAN"}


@cli.command()
def generate_table(
    pattern: str = "models/**/mix*/metadata.json",
    output: str = "results/mixed/results.csv",
    average: bool = False,
):
    df = get_df(pattern, average)
    df = df.sort_values(by="Model", ascending=True)
    df.to_csv(output, float_format="%.4f", index=False)


def get_df(input_pattern: str, average_seeds: bool, transformations: bool = False):
    paths = glob.glob(input_pattern)
    rows = []
    for path in paths:
        with open(path, "r") as f:
            try:
                metadata = json.load(f)
            except:
                print(path)
            try:
                row = {
                    "Model": MODEL[metadata["model_dir"].split("/")[1]],
                    "Rozmiar zbioru uczącego": metadata["train_files"],
                    "Błąd wal.": np.power(10, metadata["final_val_loss"]),
                    "Błąd test.": np.power(10, metadata["test_mse_loss"]),
                    "Błąd wal.[log10]": metadata["final_val_loss"],
                    "Błąd test.[log10]": metadata["test_mse_loss"],
                    "Czas uczenia[h]": metadata["training_hours"],
                    "Ziarno": metadata["random_seed"],
                }
                if row["Rozmiar zbioru uczącego"] == 0:
                    row["Rozmiar zbioru uczącego"] = 7490 - 512
                if transformations:
                    row["Typ"] = metadata["transformations"]
                rows.append(row)
            except KeyError:
                continue
    df = pd.DataFrame(rows)
    if average_seeds:
        agg_cols = ["Model", "Rozmiar zbioru uczącego"]
        if transformations:
            agg_cols.append("Typ")
        df = df.groupby(agg_cols).mean().reset_index()
        df = df.drop("Ziarno", axis=1)
        df["Błąd wal.[log10]"] = np.log10(df["Błąd wal."])
        df["Błąd test.[log10]"] = np.log10(df["Błąd test."])
    return df


if __name__ == "__main__":
    typer.run(generate_table)
