#!/usr/bin/env python3
"""
Group transcripts by the trailing index in each utt_id.

Each line of the input text file is expected to look like:
    RD_NH_051  This is the transcript for the utterance.

The script produces a JSON file whose keys are the index strings
(e.g. "051") and whose values are lists of the transcripts that share
that index.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def collect_transcripts(path: Path) -> dict[str, list[str]]:
    """Read *path* and return a dict {index: [transcripts]}."""
    groups: defaultdict[str, list[str]] = defaultdict(list)

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.rstrip()
            if not line:
                continue  # skip blank lines

            # Split once: utt_id, the rest is the transcript (may contain spaces)
            try:
                utt_id, transcript = line.split(maxsplit=1)
            except ValueError:
                raise ValueError(
                    f"Line {line_num} is malformed: “{line}”. "
                    "Each line must contain at least utt_id and transcript."
                ) from None

            # Take the substring after the final underscore in utt_id
            index = utt_id.rsplit("_", 1)[-1]
            groups[index].append(transcript)

    return groups


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Group transcripts by trailing utt_id index."
    )
    parser.add_argument("in_txt", type=Path, help="Input text file")
    parser.add_argument("out_json", type=Path, help="Output JSON file")
    args = parser.parse_args()

    grouped = collect_transcripts(args.in_txt)

    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(grouped, f, ensure_ascii=False, indent=2)

    print(f"Saved grouped transcripts to {args.out_json}")


if __name__ == "__main__":
    main()
