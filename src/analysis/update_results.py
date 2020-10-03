import json
import os
from glob import glob
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from tqdm import tqdm

from constants import (CSV_KEYS, OUTPUT_CSV, OUTPUT_MODELS_DIR, RESULTS_DIR,
                       TRAINING_RESULTS_FILENAME)

sns.set_style("darkgrid")
sns.set(font_scale=1.3)
LABEL_SIZE = 12


def update(output_csv: str = OUTPUT_CSV):
    pattern = os.path.join(OUTPUT_MODELS_DIR, "*/*", TRAINING_RESULTS_FILENAME)
    models_paths = glob(pattern)
    models_paths = [Path(path) for path in models_paths]

    rows = []
    for model_path in tqdm(models_paths, desc="Updating results"):
        result_path = Path(RESULTS_DIR).joinpath(Path(*model_path.parts[1:]))
        if not result_path.exists():
            result_path.parent.mkdir(parents=True, exist_ok=True)
            p = str(result_path)
            if "gan" not in p:
                make_plot(model_path, result_path.parent)
            else:
                make_plot_gan(model_path, result_path.parent)
            copyfile(model_path, result_path)
        model_csv = parse_json(result_path)
        rows.append(model_csv)

    df = pd.DataFrame(rows)
    df = df[CSV_KEYS]
    df.to_csv(output_csv, float_format="%.4f", index=False)


def make_plot(input_json_path: str, output_dir: Path):
    with open(input_json_path, "r") as f:
        data = json.load(f)
    steps = data["training_steps"]
    steps_per_train_loss = data["save_every_n_steps"]
    steps_per_val_loss = steps / len(data["val_losses"])

    val_losses_data = get_step_loss_pairs(
        [10 ** loss for loss in data["val_losses"]], steps_per_val_loss
    )

    train_losses_data = get_step_loss_pairs(
        data["intermediate_train_losses"], steps_per_train_loss
    )

    save_train_val_plot(
        train_losses=train_losses_data,
        val_losses=val_losses_data,
        ylabel="Błąd",
        title="Przebieg uczenia",
        output_path=output_dir.joinpath("training.png"),
        logy=False,
    )
    save_train_val_plot(
        train_losses=train_losses_data,
        val_losses=val_losses_data,
        ylabel="Błąd",
        title="Przebieg uczenia",
        output_path=output_dir.joinpath("log_training.png"),
        logy=True,
    )
    plt.close("all")


def make_plot_gan(input_json_path: str, output_dir: Path):
    with open(input_json_path, "r") as f:
        data = json.load(f)
    steps = data["training_steps"]
    steps_per_train_loss = data["save_every_n_steps"]
    steps_per_val_loss = steps / len(data["val_losses"])

    val_losses_data = get_step_loss_pairs(
        [10 ** loss for loss in data["val_losses"]], steps_per_val_loss
    )

    discriminator_real_losses = data["intermediate_train_d_real_losses"]
    discriminator_fake_losses = data["intermediate_train_d_fake_losses"]
    generator_adv_losses = data["intermediate_train_g_adv_losses"]
    generator_l1_losses = data["intermediate_train_g_l1_losses"]

    discriminator_total_losses = [
        f + r for f, r in zip(discriminator_fake_losses, discriminator_real_losses)
    ]
    generator_total_losses = [
        a + l for a, l in zip(generator_adv_losses, generator_l1_losses)
    ]

    discriminator_real_losses_data = get_step_loss_pairs(
        discriminator_real_losses, steps_per_train_loss
    )
    discriminator_fake_losses_data = get_step_loss_pairs(
        discriminator_fake_losses, steps_per_train_loss
    )
    generator_adv_losses_data = get_step_loss_pairs(
        generator_adv_losses, steps_per_train_loss
    )
    generator_l1_losses_data = get_step_loss_pairs(
        generator_l1_losses, steps_per_train_loss
    )

    discriminator_total_losses_data = get_step_loss_pairs(
        discriminator_total_losses, steps_per_train_loss
    )
    generator_total_losses_data = get_step_loss_pairs(
        generator_total_losses, steps_per_train_loss
    )

    save_adversarial_losses_plot(
        d_real_losses=discriminator_real_losses_data,
        d_fake_losses=discriminator_fake_losses_data,
        g_adv_losses=generator_adv_losses_data,
        g_l1_losses=generator_l1_losses_data,
        ylabel="Błąd",
        title="Poszczególne składowe błędu podczas uczenia",
        output_path=output_dir.joinpath("adv_losses.png"),
        logy=False,
    )

    save_adversarial_losses_plot(
        d_real_losses=discriminator_real_losses_data,
        d_fake_losses=discriminator_fake_losses_data,
        g_adv_losses=generator_adv_losses_data,
        g_l1_losses=generator_l1_losses_data,
        ylabel="Błąd",
        title="Poszczególne składowe błędu podczas uczenia",
        output_path=output_dir.joinpath("log_adv_losses.png"),
        logy=True,
    )

    save_generator_discriminator_val_loss_plot(
        discriminator_losses=discriminator_total_losses_data,
        generator_losses=generator_total_losses_data,
        val_losses=val_losses_data,
        ylabel="Błąd",
        title="Błąd walidacyjny na tle generatora i dyskryminatora",
        output_path=output_dir.joinpath("sum_val_losses.png"),
        logy=False,
    )

    save_generator_discriminator_val_loss_plot(
        discriminator_losses=discriminator_total_losses_data,
        generator_losses=generator_total_losses_data,
        val_losses=val_losses_data,
        ylabel="Błąd",
        title="Błąd walidacyjny na tle generatora i dyskryminatora",
        output_path=output_dir.joinpath("log_sum_val_losses.png"),
        logy=True,
    )
    plt.close("all")


def get_step_loss_pairs(
    losses: Sequence[float], steps_per_loss: int
) -> List[Tuple[int, float]]:
    return [
        ((i + 1) * steps_per_loss, train_loss + 10e-5)
        for i, train_loss in enumerate(losses)
    ]


def save_train_val_plot(
    train_losses: List[Tuple[int, float]],
    val_losses: List[Tuple[int, float]],
    ylabel: str,
    title: str,
    output_path: Path,
    logy: bool,
):
    plt.figure()
    _, ax = plt.subplots()

    sns.lineplot(
        [i[0] for i in train_losses],
        [i[1] for i in train_losses],
        label="Błąd treningowy",
    )
    plot = sns.lineplot(
        [i[0] for i in val_losses], [i[1] for i in val_losses], label="Błąd walidacyjny"
    )
    plot.set(xlabel="Kroki uczenia", ylabel=ylabel, title=title)
    if logy:
        plot.set(yscale="log")

    plot.set_yticklabels(plot.get_yticks(), size=LABEL_SIZE)
    plot.set_xticklabels(plot.get_xticks(), size=LABEL_SIZE)

    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())

    plot.get_figure().savefig(output_path, bbox_inches="tight")


def save_adversarial_losses_plot(
    d_real_losses: List[Tuple[int, float]],
    d_fake_losses: List[Tuple[int, float]],
    g_adv_losses: List[Tuple[int, float]],
    g_l1_losses: List[Tuple[int, float]],
    ylabel: str,
    title: str,
    output_path: Path,
    logy: bool,
):
    plt.figure()
    _, ax = plt.subplots()

    sns.lineplot(
        [i[0] for i in d_real_losses],
        [i[1] for i in d_real_losses],
        label="Błąd rzecz. [D]",
    )
    sns.lineplot(
        [i[0] for i in d_fake_losses],
        [i[1] for i in d_fake_losses],
        label="Błąd gen. [D]",
    )
    sns.lineplot(
        [i[0] for i in g_adv_losses],
        [i[1] for i in g_adv_losses],
        label="Bład przeciw. [G]",
    )
    plot = sns.lineplot(
        [i[0] for i in g_l1_losses], [i[1] for i in g_l1_losses], label="Bład L1 [G]"
    )
    plot.set(xlabel="Kroki uczenia", ylabel=ylabel, title=title)
    if logy:
        plot.set(yscale="log")

    plot.set_yticklabels(plot.get_yticks(), size=LABEL_SIZE)
    plot.set_xticklabels(plot.get_xticks(), size=LABEL_SIZE)

    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())

    plot.get_figure().savefig(output_path, bbox_inches="tight")


def save_generator_discriminator_val_loss_plot(
    generator_losses: List[Tuple[int, float]],
    discriminator_losses: List[Tuple[int, float]],
    val_losses: List[Tuple[int, float]],
    ylabel: str,
    title: str,
    output_path: Path,
    logy: bool,
):
    plt.figure()
    _, ax = plt.subplots()

    sns.lineplot(
        [i[0] for i in discriminator_losses],
        [i[1] for i in discriminator_losses],
        label="Całkowity błąd [D]",
    )
    sns.lineplot(
        [i[0] for i in generator_losses],
        [i[1] for i in generator_losses],
        label="Całkowity błąd [G]",
    )
    plot = sns.lineplot(
        [i[0] for i in val_losses], [i[1] for i in val_losses], label="Błąd walidacyjny"
    )
    plot.set(xlabel="Kroki uczenia", ylabel=ylabel, title=title)
    if logy:
        plot.set(yscale="log")

    plot.set_yticklabels(plot.get_yticks(), size=LABEL_SIZE)
    plot.set_xticklabels(plot.get_xticks(), size=LABEL_SIZE)

    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())

    plot.get_figure().savefig(output_path, bbox_inches="tight")


def parse_json(path: Path) -> Dict[str, Any]:
    with open(str(path), "r") as f:
        data = json.load(f)

    results = {}
    for key in CSV_KEYS:
        if key in data.keys():
            results[key] = data[key]
    return results


if __name__ == "__main__":
    update()
