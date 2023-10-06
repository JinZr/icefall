import argparse
import json
import logging

import lhotse
from lhotse import load_manifest_lazy
from lhotse.utils import Pathlike
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(
        description="merge translation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifests",
        type=str,
        required=True,
        help="The input manifests to be merged.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="The output merged manifest.",
    )
    parser.add_argument(
        "translations",
        type=str,
        nargs="+",
        required=True,
        help="The translation files.",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    manifests = args.manifests
    output = args.output
    translations = args.translations

    logging.info(f"Merging {translations} into {manifests}.")

    for idx, translation in enumerate(translations):
        with open(translation, "r", encoding="utf-8") as f:
            translation = json.load(f)

            version = translation["version"]
            language = translation["language"]
            audios = translation["audios"]
            segments = dict()

            for audio in tqdm(audios, desc=f"Processing {translation}."):
                segs = audio["segments"]
                for seg in segs:
                    if idx == 0:
                        segments[seg["sid"]] = {f"{language}", seg["text_raw"]}
                    else:
                        segments[seg["sid"]][f"{language}"] = seg["text_raw"]

            manifests = load_manifest_lazy(manifests)
            for manifest in tqdm(manifests, desc=f"Processing manifests."):
                manifest.custom = segments[manifest.id]

            manifests.to_file(output)
