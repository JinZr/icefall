import argparse
from pathlib import Path

from lhotse import CutSet
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_cuts", type=str, help="Path to the input cut manifest")
    parser.add_argument("output_cuts", type=str, help="Path to the output cut manifest")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    input_cuts = CutSet.from_file(args.input_cuts)
    fixed_cuts = []
    for cut in tqdm(input_cuts, desc="Fixing supervisions"):
        duration = cut.duration
        for supervision in cut.supervisions:
            if supervision.start < 0:
                supervision.start = 0
            if supervision.end > duration:
                supervision.end = duration - supervision.start
        fixed_cuts.append(cut)

    output_cuts = CutSet.from_cuts(input_cuts)
    output_cuts.to_jsonl(args.output_cuts)
    print(f"Cut manifest with supervisions fixed was saved to {args.output_cuts}")