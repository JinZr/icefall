from argparse import ArgumentParser
from pathlib import Path
from typing import List

import torch
from pyannote.audio import Pipeline
from var_path import HUGGINGFACE_ACCESS_TOKEN


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="""Path to the audio file""",
    )
    return parser.parse_args()

def diar(args, pipeline, wav_path: Path) -> List[str]:
    spkr_diar = []
    diarization = pipeline(wav_path)
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        spkr_diar.append(f"start={turn.start:.2f}s stop={turn.end:.2f}s speaker_{speaker}")
        # start=0.2s stop=1.5s speaker_0  
        # start=1.8s stop=3.9s speaker_1
        # start=4.2s stop=5.7s speaker_0
        # ...
    return spkr_diar


if __name__ == "__main__":
    args = get_args()
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HUGGINGFACE_ACCESS_TOKEN.read_text().strip(),
    )
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    
    for audio_file in args.audio_dir.glob("*.wav"):
        spkr_diar = diar(args, pipeline, args.audio_dir)
        for line in spkr_diar:
            print(line)
