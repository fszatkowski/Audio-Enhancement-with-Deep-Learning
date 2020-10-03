import pandas as pd
import typer

cli = typer.Typer()


@cli.command()
def generate_table(
    ae: str = "results/batch_norm/short_results.csv",
    bn: str = "results/batch_norm/bn_short_results.csv",
    output: str = "results/batch_norm/comp.csv",
):
    ae_df = pd.read_csv(ae, index_col=[0, 1, 2])
    bn_df = pd.read_csv(bn, index_col=[0, 1, 2])

    ae_df = ae_df.loc[bn_df.index]

    df = bn_df - ae_df
    df.to_csv(output, float_format="%.6f", index=True)


if __name__ == "__main__":
    typer.run(generate_table)
