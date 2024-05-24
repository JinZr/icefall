import argparse
import logging
from pathlib import Path

from edf_utils import edf_contents


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-edf", type=str, required=True, help="Path to the edf file."
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    input_edf = args.input_edf
    fname = Path(input_edf).stem

    logging.basicConfig(level=logging.INFO)

    _, _, header = edf_contents(input_edf)

    logging.info(
        f"{fname}\t{header['startdate'].year}-{header['startdate'].month}-{header['startdate'].day} {header['startdate'].hour}:{header['startdate'].minute}:{header['startdate'].second}\n"
    )
