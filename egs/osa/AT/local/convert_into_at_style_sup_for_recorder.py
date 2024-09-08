import argparse

from lhotse import CutSet, SupervisionSegment
from tqdm import tqdm


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
    for cut in tqdm(input_cuts, desc="Converting supervisions"):
        cut.supervisions = [
            SupervisionSegment(
                id=cut.id,
                recording_id=cut.recording.id,
                start=0,
                duration=cut.duration,
                channel=0,
                text="0",
                language="Sleep",
                speaker=cut.id.split("_")[0],
                custom={"category": "正常", "audio_event": "0"},
            )
        ]
        fixed_cuts.append(cut)

    output_cuts = CutSet.from_cuts(fixed_cuts)
    output_cuts.to_jsonl(args.output_cuts)
    print(f"Cut manifest with supervisions fixed was saved to {args.output_cuts}")
