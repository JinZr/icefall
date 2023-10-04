import argparse
import json
import logging

import lhotse
from lhotse import load_manifest_lazy
from lhotse.utils import Pathlike


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

    for translation in translations:
        with open(translation, "r", encoding="utf-8") as f:
            translation = json.load(f)

            version = translation["version"]
            language = translation["language"]
