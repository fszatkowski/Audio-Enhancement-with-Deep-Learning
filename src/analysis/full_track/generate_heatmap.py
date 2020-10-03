import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from matplotlib.colors import LogNorm

from analysis.full_track.generate_table import get_df

sns.set_style("darkgrid")
sns.set(font_scale=1.3)
cli = typer.Typer()


@cli.command()
def generate_table(
    pattern: str = "results/full_track_eval/*",
    output: str = "results/full_track/heatmap.png",
    weights: str = "Równomierny",
    target: str = "SNR",
    title: str = "Błąd średniokwadratowy",
    name: str = "model",
    lognorm: bool = False,
):
    df = get_df(pattern, name)
    df = df[df["Rodzaj wag"].str.contains(weights)]
    create_heatmap(df, output, weights, target, title, lognorm)


def create_heatmap(
    df: pd.DataFrame,
    output_path: str,
    weights: str,
    target_column: str,
    title: str,
    lognorm: bool,
):
    model_names = list(df["Model"].unique())
    model_names.remove("Wszystkie")
    model_names.append("Wszystkie")

    noise_types = list(df["Szum"].unique())
    heatmap_df = pd.DataFrame(index=model_names, columns=noise_types)
    df = df[df["Rodzaj wag"].str.contains(weights)]
    for model_name in model_names:
        for noise_type in noise_types:
            val = df[
                (df["Model"].str.contains(model_name, regex=False))
                & (df["Szum"].str.contains(noise_type, regex=False))
            ][target_column].values
            assert len(val) == 1
            heatmap_df.loc[model_name, noise_type] = val[0]
    heatmap_df = heatmap_df.astype(float)

    plt.figure(figsize=(12, 10))
    plt.tight_layout()
    if lognorm:
        plot = sns.heatmap(heatmap_df, cmap="Blues", norm=LogNorm(), annot=True)
    else:
        plot = sns.heatmap(heatmap_df, cmap="Blues", annot=True)
    plot.set(title=title, ylabel="Model", xlabel="Badany rodzaj szumu")
    plot.set_xticklabels(plot.get_xticklabels(), size=12)
    plot.set_yticklabels(plot.get_yticklabels(), size=12)
    plt.xticks(rotation=30)
    plt.yticks(rotation=30)
    plot.get_figure().savefig(output_path + ".png")
    heatmap_df.to_csv(output_path + "_abs.csv")
    if lognorm:
        heatmap_df = np.log10(heatmap_df)
        heatmap_df.to_csv(output_path + "_log.csv")
    plt.close()


if __name__ == "__main__":
    typer.run(generate_table)
