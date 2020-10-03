from argparse import ArgumentParser, Namespace
from glob import glob
from pathlib import Path

import seaborn as sns

from analysis.overfit.generate_table import get_df
from analysis.plot import plot

sns.set_style("darkgrid")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="results/overfit_segan_l1_vs_loss",
        help="Path to output plot",
    )
    parser.add_argument(
        "--abs_loss",
        default=True,
        action="store_false",
        help="Use absolute values for loss instead of using logarithm",
    )
    return parser.parse_args()


def make_plot(output_path: str, log_loss: bool = True):
    model_paths = [Path(path) for path in glob("models/segan/overfit_*/metadata.json")]
    df = get_df(model_paths)
    df["L1"] = df["L1"].astype(float)

    plot(
        function=sns.lineplot,
        output_path=output_path,
        title=f"Błąd w zależności od współczynnika L1",
        xlabel="L1",
        ylabel="Błąd",
        logy=log_loss,
        logx=True,
        yticks_format_function=lambda x, pos: "%.3f" % x,
        xticks_format_function=lambda x, pos: "%.0e" % x,
        x="L1",
        y="Błąd",
        data=df,
    )


if __name__ == "__main__":
    args = parse_args()
    make_plot(output_path=args.output_path, log_loss=args.abs_loss)
