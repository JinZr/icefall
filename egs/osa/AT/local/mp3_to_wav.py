import argparse
import logging

import lhotse
import numpy as np
import soundfile


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--output-wav", type=str, required=True, help="Path to the output wav file."
    )
    parser.add_argument(
        "--input-mp3", type=str, required=True, help="Path to the input mp3 file."
    )

    return parser


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    parser = get_parser()
    args = parser.parse_args()

    logging.info(f"Reading sound files: {args.input_mp3}")
    wav, sampling_rate = soundfile.read(args.input_mp3)

    logging.info(f"Concatenating channels")
    wav = np.average(wav, axis=-1)

    logging.info(f"Saving waveform to {args.output_wav}")
    lhotse.audio.save_audio(dest=args.output_wav, src=wav, sampling_rate=sampling_rate)
