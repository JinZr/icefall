#!/usr/bin/env python3
"""
Convert a Kaldi utt2spk file to spk2utt format.

Usage:
    python utt2spk_to_spk2utt.py input_utt2spk.txt output_spk2utt.txt
"""

import sys
from collections import defaultdict


def read_utt2spk(utt2spk_path):
    """
    Reads utt2spk file and returns a mapping from speaker to list of utterances.
    """
    spk2utts = defaultdict(list)
    with open(utt2spk_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue  # skip malformed lines
            utt_id, spk_id = parts
            spk2utts[spk_id].append(utt_id)
    return spk2utts

def write_spk2utt(spk2utts, output_path):
    """
    Writes the spk2utt mapping to a file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for spk, utts in sorted(spk2utts.items()):
            f.write(f"{spk} {' '.join(utts)}\n")

def main():
    if len(sys.argv) != 3:
        print("Usage: python utt2spk_to_spk2utt.py <input_utt2spk> <output_spk2utt>")
        sys.exit(1)

    input_utt2spk = sys.argv[1]
    output_spk2utt = sys.argv[2]

    spk2utts = read_utt2spk(input_utt2spk)
    write_spk2utt(spk2utts, output_spk2utt)
    print(f"Wrote {len(spk2utts)} speakers to '{output_spk2utt}'")

if __name__ == "__main__":
    main()