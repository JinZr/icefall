import argparse
from copy import deepcopy

from lhotse import CutSet
from lhotse.supervision import AlignmentItem
from tqdm import tqdm

DURATION = 5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-cuts", type=str, required=True, help="Path to the input cut manifest"
    )
    parser.add_argument(
        "--output-cuts", type=str, required=True, help="Path to the output cut manifest"
    )
    return parser


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()

    input_cuts = CutSet.from_file(args.input_cuts)
    fixed_cuts = []
    for cut in tqdm(input_cuts, desc="Fixing supervisions"):
        new_cut = deepcopy(cut)
        new_cut.supervisions = []
        duration = cut.duration
        all_ali = []
        for sup_index, supervision in enumerate(cut.supervisions):
            new_supervision = deepcopy(supervision)
            if sup_index == 0:
                if supervision.start < 0:
                    new_supervision.start = 0
                if supervision.duration > duration:
                    new_supervision.duration = (
                        duration
                        if len(cut.supervisions) == 1
                        else cut.supervisions[sup_index + 1].start
                    )
                if supervision.start < 0:
                    new_supervision.start = 0
            else:
                new_supervision.duration = (
                    cut.supervisions[sup_index + 1].start
                    if sup_index + 1 < len(cut.supervisions)
                    else duration - new_supervision.start
                )

            new_supervision.alignment = {
                "event": list(
                    filter(
                        lambda e: e.start >= cut.start
                        and e.start < cut.start + DURATION,
                        supervision.alignment["event"],
                    )
                ),
            }
            new_cut.supervisions.append(new_supervision)
        fixed_cuts.append(new_cut)

    output_cuts = CutSet.from_cuts(fixed_cuts)
    output_cuts.to_jsonl(args.output_cuts)
    print(f"Cut manifest with supervisions fixed was saved to {args.output_cuts}")
