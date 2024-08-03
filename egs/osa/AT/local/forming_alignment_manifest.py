import argparse
import os
from pathlib import Path

from lhotse import RecordingSet, Seconds, SupervisionSegment, SupervisionSet

LABEL_MAPPING = {
    "正常": 0,
    "低通气": 1,
    "A. 阻塞性": 2,
    "A. 中枢性": 3,
    "A. 混合性": 4,
    "单一打呼": 5,
    "鼾声串": 5,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-dir",
        type=str,
        default="./format_csv/",
        help="Path to the directory containing the CSV files",
    )
    parser.add_argument(
        "--audio-dir", type=str, help="Path to the input audio file", required=True
    )
    parser.add_argument(
        "--output-dir", type=str, help="Path to the output manifest", required=True
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    csv_dir = Path(args.csv_dir)
    audio_dir = Path(args.audio_dir)

    audio_set = RecordingSet.from_dir(
        path=audio_dir,
        pattern="*16k.wav",
        num_jobs=8,
    )

    supervision_segments = []
    for recording in audio_set.recordings:
        recording_id = recording.id
        with open(csv_dir / f"{recording_id.split('_')[0]}.csv", "r") as f:
            sup_lines = f.readlines()
        for idx, sup_line in enumerate(sup_lines):
            # line format as: 2024-03-19 21:38:45.019000,0:38:40.019000,21.465,A. 阻塞性
            if len(sup_line) == 1:
                continue
            global_time, local_time, duration, text = sup_line.strip().split(",")
            try:
                start_time_stamp, millisecond = local_time.split(".")
            except:
                start_time_stamp = local_time
            hour, minute, second = start_time_stamp.split(":")
            start_time = Seconds(
                int(hour) * 3600
                + int(minute) * 60
                + int(second)
                + float(millisecond) / 1000
            )

            duration = Seconds(float(duration))
            if float(duration) < 0.05:
                continue

            duration = float(duration)
            language = "Sleep"
            channel = 0
            try:
                segment = SupervisionSegment(
                    id=f"{recording_id}-{idx}",
                    recording_id=recording_id,
                    start=start_time,
                    duration=duration,
                    channel=channel,
                    text=f"{LABEL_MAPPING[text]}",
                    language=language,
                    speaker=f"{recording_id.split('_')[0]}",
                    custom={"category": text},
                )
                supervision_segments.append(segment)
            except:
                continue
    supervision_set = SupervisionSet.from_segments(supervision_segments)

    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    audio_set.to_jsonl(f"{args.output_dir}/osa_recordings_all.jsonl.gz")
    supervision_set.to_jsonl(f"{args.output_dir}/osa_supervisions_all.jsonl.gz")
