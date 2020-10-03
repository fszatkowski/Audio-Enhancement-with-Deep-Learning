import os
from argparse import ArgumentParser, Namespace
from random import seed, shuffle

import torch
from tqdm import tqdm

from data.utils import load_file


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input_files", "-i", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, required=True)
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--num_files", "-f", type=int, default=0)
    return parser.parse_args()


def process_dataset(output_dir: str, input_list: str, batch_size: int, max_files: int):
    with open(input_list, "r") as f:
        files = [file.strip() for file in f.readlines()]
    seed(2112)
    shuffle(files)
    if max_files:
        files = files[:max_files]
    os.makedirs(output_dir, exist_ok=True)

    batch = []
    batch_num = 0
    for file in tqdm(files, desc="Processing files"):
        y, _ = load_file(file, 44100)
        x, _ = load_file(file, 22050)

        ratio = y.shape[1] / x.shape[1]
        if ratio != 2:
            raise ValueError()

        batch.append([x, y])
        if len(batch) == batch_size:
            torch.save(batch, os.path.join(output_dir, f"batch_{batch_num}.pt"))
            batch = []
            batch_num += 1
    if len(batch) > 0:
        torch.save(batch, os.path.join(output_dir, f"batch_{batch_num}.pt"))


if __name__ == "__main__":
    args = parse_args()
    process_dataset(
        output_dir=args.output_dir,
        input_list=args.input_files,
        batch_size=args.batch_size,
        max_files=args.num_files,
    )
