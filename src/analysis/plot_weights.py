import numpy as np
import seaborn as sns
import typer

from inference.restore_weights import get_weights

cli = typer.Typer()
sns.set_style("darkgrid")
sns.set(font_scale=1.3)


@cli.command()
def plot(output: str = "weights.png", size: int = 16383):
    x = np.arange(0, size, 1)

    sns.lineplot(x, get_weights("uniform", size), label="Jednorodna")
    sns.lineplot(x, get_weights("tri", size), label="Trójkątna")
    sns.lineplot(x, get_weights("tz_1", size), label="Trapezowa (1)")
    plot = sns.lineplot(x, get_weights("tz_2", size), label="Trapezowa (2)")
    plot.set(title="Wagi wykorzystane przy łączeniu fragmentów")
    plot.get_figure().savefig(output)


if __name__ == "__main__":
    typer.run(plot)
