import argparse
import logging
from pathlib import Path

import soundfile
from edf_utils import edf_contents


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-edf", type=str, required=True, help="Path to the edf file."
    )
    parser.add_argument(
        "--output-wav", type=str, required=True, help="Path to the wav file."
    )
    return parser


def get_audio(signal_headers, signals):
    audio_header = signal_headers[-4]
    audio = signals[-4]
    return audio_header, audio


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    input_edf = args.input_edf
    output_wav = Path(args.output_wav)

    signals, signal_headers, _ = edf_contents(input_edf)
    print(signal_headers)
    exit(9)
    audio_header, audio = get_audio(signal_headers, signals)

    assert audio_header["label"] == "Audio", "Audio channel not matched"

    physical_max = audio_header["physical_max"]

    audio = audio / physical_max

    soundfile.write(f"{output_wav}", audio, 8000)
    logging.info(f"Saved {output_wav}")
