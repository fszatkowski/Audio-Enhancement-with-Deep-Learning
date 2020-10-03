import numpy as np
import seaborn as sns
import typer

from analysis.mixed.generate_table import get_df
from analysis.plot import plot

cli = typer.Typer()

MODEL = {"autoencoder": "Autoenkoder", "wavenet": "WaveNet", "segan": "SEGAN"}


@cli.command()
def plot_mixed(
    pattern: str = "models/**/mix*/metadata.json",
    output: str = "results/mixed/results.csv",
    loss: str = "val",
    logx: bool = False,
    logy: bool = False,
):
    df = get_df(pattern, False)

    if loss == "val":
        loss_col = "Błąd wal."
        title = "Błąd walidacyjny w zależności od wielkości zbioru uczącego"
    elif loss == "test":
        loss_col = "Błąd test."
        title = "Błąd testowy w zależności od wielkości zbioru uczącego"
    else:
        raise ValueError(f"Given loss type is not supported: {loss}")

    plot(
        output_path=output,
        function=sns.lineplot,
        title=title,
        xlabel="Liczba plików uczących",
        ylabel="Błąd",
        logx=logx,
        logy=logy,
        xticks=np.array([256, 512, 1024, 2048, 4096, 8192]),
        yticks=np.array([0.05, 0.04, 0.03, 0.02, 0.01, 0.005]),
        x="Rozmiar zbioru uczącego",
        y=loss_col,
        hue="Model",
        data=df,
    )


if __name__ == "__main__":
    typer.run(plot_mixed)
