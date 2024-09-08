import argparse
import random
from pathlib import Path

from lhotse import CutSet
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-cuts", type=str, required=True, help="Path to the input cut manifest"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Path to the output cut manifest"
    )
    return parser


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()

    input_cuts = CutSet.from_file(args.input_cuts)
    fixed_cuts = []
    filtered_cuts = []
    for cut in tqdm(input_cuts, desc="Fixing supervisions"):
        if len(cut.supervisions) == 0:
            filtered_cuts.append(cut)
            continue
        fixed_cuts.append(cut)

    output_cuts = CutSet.from_cuts(fixed_cuts)
    output_cuts.to_jsonl(
        Path(args.output_dir)
        / "osa_cuts_recorder_batch2_windows_fixed_filtered_empty_sup.jsonl.gz"
    )
    random.shuffle(filtered_cuts)
    filtered_cuts = CutSet.from_cuts(filtered_cuts)
    filtered_cuts.to_jsonl(
        Path(args.output_dir)
        / "osa_cuts_recorder_batch2_windows_fixed_should_be_normal.jsonl.gz"
    )
    print(f"Cut manifest with supervisions fixed was saved to {args.output_dir}")
