import argparse
import logging
import os
from pathlib import Path

from lhotse import Recording, RecordingSet, Seconds, SupervisionSegment, SupervisionSet

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
        "--csv-path",
        type=str,
        help="Path to the CSV files",
        required=True,
    )
    parser.add_argument(
        "--audio-path", type=str, help="Path to the input audio file", required=True
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to the directory to save the output manifest",
        required=True,
    )
    parser.add_argument(
        "--offset",
        type=int,
        required=True,
        help="Offset in seconds to apply to the supervision segments",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.info(args)

    csv_path = Path(args.csv_path)
    audio_path = Path(args.audio_path)
    offset = Seconds(args.offset)

    speaker_id = csv_path.stem

    recording = Recording.from_file(
        path=audio_path,
        recording_id=speaker_id,
        force_read_audio=True,
    )

    supervision_segments = []
    recording_id = recording.id
    with open(csv_path, "r") as f:
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

        if start_time + duration < offset:
            continue

        if start_time < offset:
            start_time = Seconds(0)
            duration -= offset
        else:
            start_time -= offset
            duration -= offset

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
                speaker=recording_id,
                custom={"category": text},
            )
            supervision_segments.append(segment)
        except:
            continue

    supervision_set = SupervisionSet.from_segments(supervision_segments)
    audio_set = RecordingSet.from_recordings([recording])

    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    audio_set.to_jsonl(f"{args.output_dir}/osa_recordings_{speaker_id}.jsonl.gz")
    supervision_set.to_jsonl(
        f"{args.output_dir}/osa_supervisions_{{speaker_id}}.jsonl.gz"
    )
