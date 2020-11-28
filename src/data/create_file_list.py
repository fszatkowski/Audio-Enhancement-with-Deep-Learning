import os
from argparse import ArgumentParser, Namespace
from typing import Sequence

from tqdm import tqdm

from constants import RAW_DATASET_FILELIST
from data.utils import get_mp3_files, load_file


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--output_file", "-o", type=str, default=RAW_DATASET_FILELIST)
    parser.add_argument("--n_files", "-n", type=int, default=0)
    return parser.parse_args()


def create_file_list(files: Sequence[str], n_files: int, output_file_name: str):
    """ Create text file with names of relevant audio files
    (meaning those files have 2 channels, 44100 original sampling rate and produce no errors during loading
    Those files are used for training base models """
    if os.path.exists(output_file_name):
        os.remove(output_file_name)
        os.mknod(output_file_name)
    correct = 0

    if n_files:
        files = files[:n_files]

    with open(output_file_name, "a+") as f:
        for file in tqdm(files):
            try:
                audio, sr = load_file(file, sample_rate=None)
                if audio.shape[0] == 2 and sr == 44100:
                    f.write(f"{file}\n")
                    correct += 1
            except Exception as e:
                print(e)
                continue
    print(f"Checked {len(files)} files, {correct} correct.")


if __name__ == "__main__":
    args = parse_args()
    files = get_mp3_files()
    create_file_list(files, n_files=args.n_files, output_file_name=args.output_file)
