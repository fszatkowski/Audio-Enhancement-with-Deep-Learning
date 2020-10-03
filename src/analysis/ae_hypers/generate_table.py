import glob
import json

import numpy as np
import pandas as pd
import typer

cli = typer.Typer()


@cli.command()
def generate_table(
    pattern: str = "models/autoencoder/hiper*/metadata.json",
    output: str = "results/hyperparams/results.csv",
    average: bool = False,
    short: bool = False,
    activation: bool = False,
    trainhypers: bool = False,
):
    df = get_df(pattern, average, activation, trainhypers)
    df = df.sort_values(by="Błąd test.[log10]", ascending=True)
    if short:
        df.drop(columns=["Czas uczenia[h]", "Rozmiar zbioru uczącego"], inplace=True)
    df.to_csv(output, float_format="%.6f", index=False)


def get_df(
    input_pattern: str, average_seeds: bool, activation: bool, trainhypers: bool
):
    paths = glob.glob(input_pattern)
    rows = []
    for path in paths:
        with open(path, "r") as f:
            metadata = json.load(f)
            try:
                row = {
                    "Liczba warstw": metadata["num_layers"],
                    "Liczba kanałów": metadata["channels"],
                    "Rozmiar filtru": metadata["kernel_size"],
                    "Błąd wal.": np.power(10, metadata["final_val_loss"]),
                    "Błąd test.": np.power(10, metadata["test_mse_loss"]),
                    "Błąd wal.[log10]": metadata["final_val_loss"],
                    "Błąd test.[log10]": metadata["test_mse_loss"],
                    "Liczba parametrów": metadata["num_params"],
                    "Czas uczenia[h]": metadata["training_hours"],
                    "Rozmiar zbioru uczącego": metadata["train_files"],
                    "Ziarno": metadata["random_seed"],
                }
                if activation:
                    row["Funkcja aktywacji"] = metadata["activation"]
                if trainhypers:
                    row["Współczynnik uczenia"] = metadata["learning_rate"]
                    row["Wielkość batcha"] = metadata["batch_size"]
                rows.append(row)
            except KeyError:
                continue
    df = pd.DataFrame(rows)
    if average_seeds:
        hyper_cols = ["Liczba warstw", "Liczba kanałów", "Rozmiar filtru"]
        if activation:
            hyper_cols.append("Funkcja aktywacji")
        if trainhypers:
            hyper_cols.extend(["Współczynnik uczenia", "Wielkość batcha"])
        df = df.groupby(hyper_cols).mean().reset_index()
        df = df.drop("Ziarno", axis=1)
        df["Błąd wal.[log10]"] = np.log10(df["Błąd wal."])
        df["Błąd test.[log10]"] = np.log10(df["Błąd test."])
    df["Rozmiar zbioru uczącego"] = df["Rozmiar zbioru uczącego"].astype(int)
    df["Liczba parametrów"] = df["Liczba parametrów"].astype(int)

    return df


if __name__ == "__main__":
    typer.run(generate_table)
