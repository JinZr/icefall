import argparse
import logging
from pathlib import Path

from edf_utils import edf_duration, edf_start_time


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-edf", type=str, required=True, help="Path to the edf file."
    )
    parser.add_argument("--output-dir", type=str, default="./edf_date")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    input_edf = args.input_edf
    output_dir = args.output_dir
    fname = Path(input_edf).stem
    output_dir = Path(output_dir)

    if not output_dir.exists():
        output_dir.mkdir()

    logging.basicConfig(level=logging.INFO)

    start_time = edf_start_time(input_edf)
    duration = edf_duration(input_edf) / 60 / 60  # seconds --> hours

    logging.info(
        f"{fname}\t{start_time.year}-{start_time.month}-{start_time.day} {start_time.hour}:{start_time.minute}:{start_time.second}.{start_time.microsecond}\t{duration} hours"
    )
    with open(output_dir / f"{fname}.txt", "w") as fout:
        fout.write(
            f"{start_time.year}-{start_time.month}-{start_time.day} {start_time.hour}:{start_time.minute}:{start_time.second}.{start_time.microsecond}"
        )
