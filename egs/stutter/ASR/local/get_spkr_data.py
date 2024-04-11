import argparse
import logging

from lhotse import load_manifest_lazy
from lhotse.cut import Cut


def get_parser():
    parser = argparse.ArgumentParser(
        description="Get indicated speaker data for stuttering speech challenge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        default="./data/fbank/stutter_cuts_train.jsonl.gz",
        type=str,
        help="the lhotse manifest for stuttering speech challenge",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/fbank/",
        type=str,
        help="the directory for storing filtered manifest",
    )
    parser.add_argument(
        "--speaker",
        default="0037",
        type=str,
        help="the indicated speaker",
    )
    return parser


def remove_utt_of_speaker(speaker: str):
    def filter_cut(c: Cut):
        if c.supervisions[0].speaker == speaker:
            return False
        else:
            return True

    return filter_cut


def retain_utt_of_speaker(speaker: str):
    def filter_cut(c: Cut):
        if c.supervisions[0].speaker == speaker:
            return True
        else:
            return False

    return filter_cut


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    speaker = args.speaker
    manifest = args.manifest
    output_dir = args.output_dir

    manifest = load_manifest_lazy(manifest)

    new_train = manifest.filter(remove_utt_of_speaker(speaker))

    target_speaker = manifest.filter(retain_utt_of_speaker(speaker))

    new_train.to_jsonl(f"{output_dir}/stutter_cuts_train_filtered_{speaker}.jsonl.gz")
    target_speaker.to_jsonl(f"{output_dir}/stutter_cuts_{speaker}.jsonl.gz")

    logging.info(f"Filtered manifest for speaker {speaker} is saved in {output_dir}")
