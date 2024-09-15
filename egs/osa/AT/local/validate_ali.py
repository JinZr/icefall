import argparse

from lhotse import CutSet, load_manifest_lazy
from lhotse.supervision import AlignmentItem, SupervisionSegment
from tqdm import tqdm

DURATION = 5


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-cuts", type=str, required=True, help="Path to the input cut manifest"
    )
    parser.add_argument(
        "--output-cuts", type=str, required=True, help="Path to the output cut manifest"
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    input_cuts = load_manifest_lazy(args.input_cuts)
    fixed_cuts = []
    for cut in tqdm(input_cuts, desc="Validating supervisions"):
        duration = cut.duration
        ali = dict()
        for sup_index, supervision in enumerate(cut.supervisions):
            # ali += supervision.alignment["event"]
            # ali[] = supervision.alignment["event"]
            for event in supervision.alignment["event"]:
                ali[int(event.start)] = event

        if not (len(ali) == 5 or len(ali) == 0):
            idx_idx = 0
            audio_event = ""
            ali_keys = sorted(ali.keys())
            for seg_idx in range(int(cut.start), int(min(ali_keys))):
                audio_event += "0"
                audio_event += ";"
                idx_idx += 1
            for seg_idx in range(int(cut.start), int(cut.start + DURATION)):
                if seg_idx in ali_keys:
                    audio_event += ali[seg_idx].symbol
                    audio_event += ";"
                    idx_idx += 1
            for seg_idx in range(int(DURATION - idx_idx)):
                audio_event += "0"
                audio_event += ";"
                idx_idx += 1

            assert (
                len(audio_event.split(";")) == DURATION + 1
            ), f"idx_idx: {idx_idx}, cut.duration: {cut.duration}, {audio_event}, {cut}"
            cut.supervisions = [cut.supervisions[0]]
            cut.supervisions[0].audio_event = audio_event[:-1]
        else:
            if len(cut.supervisions) == 0:
                cut.supervisions.append(
                    SupervisionSegment(
                        id=cut.id,
                        start=0,
                        duration=cut.duration,
                        recording_id=cut.recording.id,
                        channel=0,
                        text="",
                        language="",
                        speaker="",
                        custom={"audio_event": "0;0;0;0;0"},
                    ))
            else:
                cut.supervisions = [cut.supervisions[0]]
                cut.supervisions[0].audio_event = ";".join(
                    [e.symbol for e in cut.supervisions[0].alignment["event"]]
                )
        if cut.duration == DURATION:
            fixed_cuts.append(cut)
    print(len(fixed_cuts))
    print(len(input_cuts))
    fixed_cutset = CutSet.from_cuts(fixed_cuts)
    fixed_cutset.to_jsonl(args.output_cuts)
