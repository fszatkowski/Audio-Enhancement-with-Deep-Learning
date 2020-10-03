import numpy as np
import seaborn as sns
import typer

sns.set_style("darkgrid")
sns.set(font_scale=1.3)


def prelu(x: np.array, alpha: float) -> np.array:
    return np.maximum(0, x) + alpha * np.minimum(0, x)


def elu(x: np.array) -> np.array:
    return np.maximum(0, x) + np.minimum(0, np.exp(x) - 1)


def gelu(x: np.array) -> np.array:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.44715 * x ** 3)))


def swish(x: np.array) -> np.array:
    return x * (1 + np.exp(-x)) ** (-1)


cli = typer.Typer()


@cli.command()
def plot(output: str = "activations.png"):
    x = np.arange(-3, 3, 0.001)

    sns.lineplot(x, prelu(x, 0), label="ReLU")
    sns.lineplot(x, prelu(x, 0.05), label="PReLU")
    sns.lineplot(x, elu(x), label="ELU")
    sns.lineplot(x, gelu(x), label="GELU")
    plot = sns.lineplot(x, swish(x), label="Swish")
    plot.set(title="Różne funkcje aktywacji")
    plot.get_figure().savefig(output)


if __name__ == "__main__":
    typer.run(plot)
