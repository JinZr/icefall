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
        "sound_files",
        type=str,
        nargs="+",
        help="The input sound file(s) to transcribe. "
        "Supported formats are those supported by torchaudio.load(). "
        "For example, wav and flac are supported. "
        "The sample rate has to be 16kHz.",
    )

    return parser


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    parser = get_parser()
    args = parser.parse_args()

    logging.info(f"Reading sound files: {args.sound_files}")
    src = []
    for sound_file in args.sound_files:
        logging.info(f"Reading {sound_file}")
        wav, sampling_rate = soundfile.read(sound_file)
        src.extend(wav)
    src = np.average(src, axis=-1)

    logging.info(f"Saving concatenated waveform to {args.output_wav}")
    lhotse.audio.save_audio(dest=args.output_wav, src=src, sampling_rate=sampling_rate)
