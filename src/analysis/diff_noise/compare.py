from argparse import ArgumentParser, Namespace
from glob import glob
from pathlib import Path

import pandas as pd

from analysis.diff_noise.generate_table import ALIASES, get_df


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--first_pattern",
        type=str,
        default="models/**/small_model_*/metadata.json",
        help="Pattern for glob to find data",
    )
    parser.add_argument(
        "-s",
        "--second_pattern",
        type=str,
        default="models/**/med_model_*/metadata.json",
        help="Pattern for glob to find data",
    )
    parser.add_argument(
        "-l", "--loss_type", type=str, required=True, help="val or test"
    )
    parser.add_argument(
        "-o", "--output_path", type=str, required=True, help="Path to output plot"
    )

    return parser.parse_args()


def main(pattern1: str, pattern2: str, loss_type: str, output_path: str):
    df1 = get_parsed_df(pattern1, loss_type)
    df2 = get_parsed_df(pattern2, loss_type)
    diff = df2 - df1
    diff.to_csv(output_path, float_format="%.4f")


def get_parsed_df(pattern: str, loss_type: str) -> pd.DataFrame:
    model_paths = [Path(path) for path in glob(pattern)]

    df = get_df(model_paths, loss_type)
    df.index = df["Model"]
    df = df[ALIASES.values()].astype(float)

    return df.loc[["Autoenkoder", "SEGAN"], :]


if __name__ == "__main__":
    args = parse_args()
    main(
        pattern1=args.first_pattern,
        pattern2=args.second_pattern,
        output_path=args.output_path,
        loss_type=args.loss_type,
    )
