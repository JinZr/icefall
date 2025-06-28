#!/usr/bin/env python
# convert_kaldi_to_csv.py
#
# Usage:
#   python convert_kaldi_to_csv.py /path/to/kaldi_data_dir --out kaldi.csv
#
# The Kaldi directory must contain:
#   wav.scp    lines: <utt_id>  <wav path | command>
#   text       lines: <utt_id>  <transcript>
#   utt2spk    lines: <utt_id>  <speaker_id>

import argparse
import csv
import sys
from pathlib import Path


def read_kaldi_mapping(file_path, key_first=True):
    """Return dict keyed by the first column."""
    mapping = {}
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(maxsplit=1)
            if len(parts) != 2:
                continue
            key, val = parts if key_first else parts[::-1]
            mapping[key] = val
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Convert Kaldi data dir to CSV.")
    parser.add_argument("data_dir", type=Path, help="Kaldi data directory")
    parser.add_argument("--out", type=Path, default="kaldi.csv", help="CSV output path")
    args = parser.parse_args()

    data_dir = args.data_dir
    wav_scp = data_dir / "wav.scp"
    text_f = data_dir / "text"
    utt2spk = data_dir / "utt2spk"

    if not all(p.exists() for p in (wav_scp, text_f, utt2spk)):
        sys.exit("Error: wav.scp, text, utt2spk must all exist in the given directory.")

    wav_map = read_kaldi_mapping(wav_scp)
    text_map = read_kaldi_mapping(text_f)
    spk_map = read_kaldi_mapping(utt2spk)

    utt_ids = sorted(set(wav_map) & set(text_map) & set(spk_map))
    if not utt_ids:
        sys.exit("No common utterance IDs across the three files.")

    with args.out.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # writer.writerow(["id", "speaker", "path", "text"])
        for uid in utt_ids:
            writer.writerow([uid, spk_map[uid], wav_map[uid], text_map[uid]])

    print(f"✓ Wrote {len(utt_ids)} rows to {args.out}")


if __name__ == "__main__":
    main()
