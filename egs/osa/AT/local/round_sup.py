import argparse
import logging

from lhotse import load_manifest
from lhotse.supervision import AlignmentItem, SupervisionSet
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sup", type=str, required=True, help="Supervision set")
    parser.add_argument(
        "--output", type=str, required=True, help="Output ali rspecifier"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    cutset: SupervisionSet = load_manifest(args.sup)
    logging.info(f"Loaded {len(cutset)} supervisions from {args.sup}")

    rounded = []
    for cut in tqdm(cutset):
        alignment = {"event": []}
        if round(cut.duration) > 0:
            cut.duration = int(round(cut.duration))
            for start in range(int(cut.duration)):
                alignment["event"].append(
                    AlignmentItem(
                        symbol=cut.text,
                        start=cut.start + start,
                        duration=1.0,
                    )
                )
            cut.alignment = alignment
            rounded.append(cut)
        else:
            continue

    SupervisionSet(segments=rounded).to_jsonl(args.output)
    logging.info(f"Rounded {len(rounded)} supervisions and saved to {args.output}")
