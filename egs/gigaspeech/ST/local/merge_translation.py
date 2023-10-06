import argparse
import json
import logging

from lhotse import CutSet, load_manifest_lazy
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
        help="The translation files.",
    )
    return parser


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

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

            logging.info(f"Processing {language}.")
            cntr = 0
            for audio in tqdm(audios):
                segs = audio["segments"]
                for seg in segs:
                    cntr += 1
                    if idx == 0:
                        segments[seg["sid"]] = {
                            f"{language}": seg["text_raw"],
                        }
                    else:
                        segments[seg["sid"]][f"{language}"] = seg["text_raw"]
            logging.info(f"Processed {cntr} segments.")
    logging.info(f"Loading {manifests}.")
    manifests = load_manifest_lazy(manifests)
    cuts = dict()
    err_cntr = 0
    for cut in tqdm(manifests):
        try:
            cut.custom = segments[cut.supervisions[0].id]
            cuts[cut.id] = cut
        except KeyError:
            err_cntr += 1
            continue
    logging.info(f"Failed to find {err_cntr} segments.")
    CutSet(cuts=cuts).to_file(output)
