import numpy as np
import seaborn as sns
import typer

from analysis.ae_hypers.generate_table import get_df
from analysis.plot import plot

cli = typer.Typer()


@cli.command()
def create_plot(
    pattern: str = "models/autoencoder/hiper*/metadata.json",
    output: str = "results/hyperparams/results.csv",
    loss: str = "val",
    logy: bool = False,
    logx: bool = False,
    average: bool = False,
):
    df = get_df(pattern, average)
    df = df[df["Błąd wal."] < 0.005]
    df["Błąd wal."] = 10 ** 4 * df["Błąd wal."]
    df["Błąd test."] = 10 ** 4 * df["Błąd test."]

    if loss == "val":
        loss_col = "Błąd wal."
        title = "Błąd walidacyjny w zależności od liczby parametrów"
    elif loss == "test":
        loss_col = "Błąd test."
        title = "Błąd testowy w zależności od liczby parametrów"
    else:
        raise ValueError(f"Loss type: {loss} should be either 'val' or 'test'")

    plot(
        output_path=output,
        function=sns.scatterplot,
        title=title,
        xlabel="Liczba parametrów",
        ylabel="Błąd [10e-4]",
        logx=logx,
        logy=logy,
        xticks=np.array([10e3, 10e4, 10e5, 10e6, 10e7, 10e8]),
        yticks=np.array([1, 3, 10, 30]),
        xticks_format_function=lambda x, pos: "%.0e" % x,
        x="Liczba parametrów",
        y=loss_col,
        data=df,
    )


if __name__ == "__main__":
    typer.run(create_plot)
