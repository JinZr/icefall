import argparse
import logging
import os
from pathlib import Path

import datasets
import torch
from datasets import load_dataset
from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet
from tqdm import tqdm

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def format_lhotse_cuts(
    dataset: datasets.dataset_dict.DatasetDict,
    subset: str = "train",
    wav_dir: str = "download/wav",
    cache_dir: str = "data/cache",
    output_dir: str = "data/manifests",
):

    def get_tts_conversation_id(wav_dir: str):
        return os.listdir(wav_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tts_conversation_id = get_tts_conversation_id(wav_dir)

    iterable_dataset = (
        dataset[subset]
        .select(range(100000))
        .filter(
            lambda x: x["conversation_id"] in tts_conversation_id,
            cache_file_name=f"{cache_dir}/filtered",
        )
    )

    recordings = []
    supervisions = []

    for row in tqdm(iterable_dataset, desc=f"Processing {subset}"):
        turn = row["turn"]
        model = row["model"]
        language = row["language"]
        conversation_id = row["conversation_id"]

        wav_path = os.path.join(wav_dir, conversation_id)

        user_index = 0
        for index, conversation in enumerate(row["conversation"]):
            if turn["role"] == "user":
                cut_id = f"{conversation_id}-{user_index}"
                recordings.append(
                    Recording.from_file(
                        path=f"{wav_path}/user_{user_index}.mp3",
                        recording_id=cut_id,
                    )
                )
                prev_text = " ".join(
                    [
                        conversation["content"]
                        for conversation in row["conversation"][:index]
                    ]
                )
                supervisions.append(
                    SupervisionSegment(
                        id=cut_id,
                        recording_id=cut_id,
                        start=0.0,
                        duration=recordings[-1].duration,
                        channel=0,
                        language=language,
                        speaker=model,
                        text=conversation["content"],
                        custom={"prev_text": prev_text},
                    )
                )
                user_index += 1
            else:
                continue

    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)

    recording_set.to_json(output_dir / f"lmsys_recordings_{subset}.json")
    supervision_set.to_json(output_dir / f"lmsys_supervisions_{subset}.json")

    logging.info(
        f"Saved {len(recordings)} recordings and {len(supervisions)} supervisions to {output_dir}"
    )


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/manifests",
        help="""Output directory to save the manifests""",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache",
        help="""Directory to cache the filtered dataset""",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="lmsys/lmsys-chat-1m",
        help="""Huggingface dataset to format""",
    )

    parser.add_argument(
        "--subset",
        type=str,
        default="train",
        help="""Subset to format""",
    )

    parser.add_argument(
        "--wav-dir",
        type=str,
        default="./download/wav",
        help="""Directory to saved TTS audio files""",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    logging.info(args)

    dataset = load_dataset(args.dataset)

    format_lhotse_cuts(
        dataset=dataset,
        subset=args.subset,
        wav_dir=args.wav_dir,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )
