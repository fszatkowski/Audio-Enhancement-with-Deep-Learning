import glob

import numpy as np
import pandas as pd
import typer

from analysis.diff_noise.generate_table import ALIASES

cli = typer.Typer()

WEIGHTS = ["uniform", "tri", "tz_1", "tz_2"]
WEIGHT_ALIASES = {
    "tri": "Trójkątny",
    "uniform": "Równomierny",
    "tz_1": "Trapezowy (1)",
    "tz_2": "Trapezowy (2)",
}


@cli.command()
def generate_table(
    pattern: str = "results/full_track_eval/*",
    output: str = "results/full_track/weights.csv",
):
    df = get_df(pattern, "none_model", True)
    df.to_csv(output, float_format="%.6f", index=False)


def get_df(input_pattern: str, column_filter: str = "model", drop_name: bool = False):
    paths = glob.glob(input_pattern)
    rows = []
    for path in paths:
        with open(path, "r") as f:
            weights = None
            for w in WEIGHTS:
                if path.endswith(w):
                    weights = w
            model = path.split("/")[-1][: -(len(weights) + 1)]
            data = pd.read_csv(path)
            data["model"] = model
            data["weights"] = weights
            data = data[data["name"].str.contains(column_filter)]
            if column_filter == "benchmark" or column_filter == "model":
                data["name"] = data["name"].str.slice(0, -(len(column_filter) + 1))
                data["name"] = data["name"].str.replace("zero_005", "zero")
                data["name"] = data["name"].apply(lambda x: ALIASES[x])
            rows.append(data)
    df = pd.concat(rows, axis=0)
    if drop_name:
        df = df.drop(columns=["name"])
        cols_out_aliases = [
            "Model",
            "Rodzaj wag",
            "MSE",
            "Log10(MSE)",
            "MAE",
            "Log10(MAE)",
            "SNR",
            "PSNR",
        ]
        cols_out = [
            "model",
            "weights",
            "mse",
            "log_mse",
            "mae",
            "log_mae",
            "snr",
            "psnr",
        ]
    else:
        cols_out_aliases = [
            "Model",
            "Szum",
            "Rodzaj wag",
            "MSE",
            "Log10(MSE)",
            "MAE",
            "Log10(MAE)",
            "SNR",
            "PSNR",
        ]
        cols_out = [
            "model",
            "name",
            "weights",
            "mse",
            "log_mse",
            "mae",
            "log_mae",
            "snr",
            "psnr",
        ]
    df["log_mse"] = np.log10(df["mse"])
    df["log_mae"] = np.log10(df["mae"])

    out = pd.DataFrame()
    out[cols_out_aliases] = df[cols_out]
    out["Model"] = out["Model"].apply(lambda x: ALIASES[x])
    out["Rodzaj wag"] = out["Rodzaj wag"].apply(lambda x: WEIGHT_ALIASES[x])
    out = out.sort_values(by=["Model", "Rodzaj wag"])
    return out


if __name__ == "__main__":
    typer.run(generate_table)
