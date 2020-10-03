import pandas as pd
import typer

from analysis.mixed.generate_table import get_df

cli = typer.Typer()


@cli.command()
def generate_table(
    pattern: str = "models/**/mix*/metadata.json",
    output: str = "results/mixed/results.csv",
    files: int = 1024,
    set: str = "val",
    log: bool = False,
):
    df = get_df(pattern, True, True)
    df = df[df["Rozmiar zbioru uczącego"] == files]

    columns = ["Model", "Typ"]
    if set == "val":
        if log:
            loss_col = "Błąd wal.[log10]"
        else:
            loss_col = "Błąd wal."
    elif set == "test":
        if log:
            loss_col = "Błąd test.[log10]"
        else:
            loss_col = "Błąd test."
    else:
        raise ValueError(f"Unknown dataset type: {set}")
    columns.append(loss_col)

    df = df[columns]
    noise_types = list(df["Typ"].unique())

    result_df = pd.DataFrame()
    for type in noise_types:
        result_df[type] = (
            df[df["Typ"] == type].drop(["Typ"], axis=1).set_index("Model")[loss_col]
        )
    result_df = result_df[["mix", "mix_weaker_zero", "mix_no_zero"]]
    result_df.columns = ["Zerowanie 1%", "Zerowanie 0.1%", "Brak zerowania"]
    result_df.to_csv(output, float_format="%.4f", index=True)


if __name__ == "__main__":
    typer.run(generate_table)
