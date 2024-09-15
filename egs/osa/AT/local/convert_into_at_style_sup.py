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
        if len(cut.supervisions) > 1:
            raise ValueError(
                f"This script expects cuts with a single supervision. {cut}"
            )
            at_label = ";".join([s.text for s in cut.supervisions])
            at_label_text = ";".join([s.custom["category"] for s in cut.supervisions])
        else:
            # at_label = cut.supervisions[0].text
            at_label = ";".join(
                [e.symbol for e in cut.supervisions[0].alignment["event"]]
            )
            at_label_text = cut.supervisions[0].custom["category"]
        # cut.custom = {"audio_event": at_label}
        cut.supervisions = [
            SupervisionSegment(
                id=cut.supervisions[0].id,
                recording_id=cut.supervisions[0].recording_id,
                start=0,
                duration=cut.duration,
                channel=cut.supervisions[0].channel,
                text=at_label,
                language="Sleep",
                speaker=cut.supervisions[0].speaker,
                alignment=cut.supervisions[0].alignment,
                custom={"category": at_label_text, "audio_event": at_label},
            )
        ]
        fixed_cuts.append(cut)

    output_cuts = CutSet.from_cuts(fixed_cuts)
    output_cuts.to_jsonl(args.output_cuts)
    print(f"Cut manifest with supervisions fixed was saved to {args.output_cuts}")
