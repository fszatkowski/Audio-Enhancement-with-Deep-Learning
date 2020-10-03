from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter, ScalarFormatter

sns.set_style("darkgrid")
sns.set(font_scale=1.3)

LABEL_FONT_SIZE = 12


def plot(
    output_path: str,
    function: Callable,
    title: str,
    xlabel: str,
    ylabel: str,
    logx: bool = False,
    logy: bool = False,
    xtext: bool = False,
    xticks: Optional[np.array] = None,
    yticks: Optional[np.array] = None,
    xticks_format_function: Optional[Callable] = None,
    yticks_format_function: Optional[Callable] = None,
    **kwargs
):
    plt.figure()
    _, ax = plt.subplots()
    plot = function(**kwargs)

    if logx:
        plot.set(xscale="log")
        if xticks is not None:
            ax.set_xticks(xticks)
            plot.set_xticklabels(plot.get_xticks(), size=LABEL_FONT_SIZE)

    if xticks_format_function is None:
        if not xtext:
            ax.xaxis.set_major_formatter(ScalarFormatter())
        else:
            plot.set_xticklabels(plot.get_xticklabels(), size=LABEL_FONT_SIZE)
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(xticks_format_function))

    if logy:
        plot.set(yscale="log")
        if yticks is not None:
            ax.set_yticks(yticks)
    plot.set_yticklabels(plot.get_yticks(), size=LABEL_FONT_SIZE)

    if yticks_format_function is None:
        ax.yaxis.set_major_formatter(ScalarFormatter())
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(yticks_format_function))

    plot.set(title=title, xlabel=xlabel, ylabel=ylabel)
    plot.get_figure().savefig(output_path, bbox_inches="tight")
