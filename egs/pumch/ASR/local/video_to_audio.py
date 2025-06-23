import os
from argparse import ArgumentParser
from pathlib import Path

import librosa
import soundfile as sf
from tqdm import tqdm


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--mov-dir",
        type=Path,
        required=True,
        help="""Path to the directory containing the video files""",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="""Path to the directory where the audio files will be saved""",
    )
    return parser.parse_args()


def main(args):
    os.makedirs(args.audio_dir, exist_ok=True)
    for video_file in tqdm(
        list(args.mov_dir.glob("*.MOV"))
        + list(args.mov_dir.glob("*.mov"))
        + list(args.mov_dir.glob("*.MP4"))
        + list(args.mov_dir.glob("*.mp4"))
    ):
        audio, sr = librosa.load(str(video_file), sr=None)
        audio_file = args.audio_dir / (video_file.stem + ".wav")
        sf.write(audio_file, audio, sr)


if __name__ == "__main__":
    args = get_args()
    main(args=args)
