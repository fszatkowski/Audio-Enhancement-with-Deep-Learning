import json
from argparse import ArgumentParser, Namespace

from inference.audio_processor import AudioProcessor


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--metadata",
        type=str,
        required=True,
        help="Path to trained model metadata",
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        required=True,
        help="Path to audio file to upsample",
    )
    parser.add_argument(
        "-o", "--output_file", type=str, required=True, help="Path to save output_file"
    )
    parser.add_argument(
        "-n",
        "--noise",
        type=str,
        default=None,
        help="Optional transformation to use on file. "
        "The name should correspond to the ones used during training "
        "(eg. TranformationsManager.get_transformations() keys).",
    )
    parser.add_argument(
        "-s",
        "--sr",
        type=int,
        default=0,
        help="If provided, "
        "will load file with this sampling rate "
        "instead of keeping original sampling rate.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.metadata, "r") as f:
        metadata = json.load(f)
    processor = AudioProcessor(metadata)

    processor.process_file(
        args.input_file,
        input_file_sr=args.sr,
        output_file_path=args.output_file,
        noise=args.noise,
    )
