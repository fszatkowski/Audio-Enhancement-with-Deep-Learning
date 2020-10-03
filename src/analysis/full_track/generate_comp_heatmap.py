import numpy as np
import seaborn as sns
import typer

from analysis.full_track.generate_heatmap import create_heatmap
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
    relative: bool = False,
):
    model_df = get_df(pattern, "model")
    benchmark_df = get_df(pattern, "benchmark")
    model_df = model_df[model_df["Rodzaj wag"].str.contains(weights)]
    benchmark_df = benchmark_df[benchmark_df["Rodzaj wag"].str.contains(weights)]
    model_df = model_df.set_index(keys=["Model", "Szum"], drop=False)
    benchmark_df = benchmark_df.set_index(keys=["Model", "Szum"], drop=False)

    numeric_cols = model_df.select_dtypes(include=np.number).columns.tolist()
    diff = model_df.copy(deep=True)

    if relative:
        diff[numeric_cols] = model_df[numeric_cols] / benchmark_df[numeric_cols]
    else:
        diff[numeric_cols] = model_df[numeric_cols] - benchmark_df[numeric_cols]
    create_heatmap(diff, output, weights, target, title, lognorm=False)


if __name__ == "__main__":
    typer.run(generate_table)
