#!/usr/bin/env python3
"""
merge_and_kfold.py

Merge multiple Kaldi data dirs, then create 5-fold speaker-level
partitions with “INV”/“SPEAKER” speakers kept in train for every fold.

Example
-------
python merge_and_kfold.py \
    --out-root data_cv5 \
    data/train_dir1 data/train_dir2 data/train_dir3
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import random
import shutil
import sys
from typing import Dict, List, Set, Tuple

KALDI_FILES_PRIORITY = (
    "wav.scp",
    "segments",
    "text",
    "utt2spk",
    "spk2gender",
    "reco2file_and_channel",
    "utt2lang",
    "spk2lang",
)  # merge these if they exist

SPECIAL_SUBSTRS = ("INV", "SPEAKER")  # hard-coded per requirement


# --------------------------------------------------------------------- #
#                    Utility helpers
# --------------------------------------------------------------------- #


def read_kaldi_kv(path: pathlib.Path) -> Dict[str, str]:
    """Read a Kaldi *two-column* text file into an ordered dictionary."""
    d = collections.OrderedDict()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            key, *vals = line.rstrip("\n").split(maxsplit=1)
            d[key] = vals[0] if vals else ""
    return d


def write_kaldi_kv(d: Dict[str, str], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for k, v in d.items():
            f.write(f"{k} {v}\n")


def make_spk2utt(utt2spk: Dict[str, str]) -> Dict[str, str]:
    spk2utt: Dict[str, List[str]] = collections.defaultdict(list)
    for u, s in utt2spk.items():
        spk2utt[s].append(u)
    # join utt lists
    return {s: " ".join(sorted(utts)) for s, utts in spk2utt.items()}


# --------------------------------------------------------------------- #
#                    Merge phase
# --------------------------------------------------------------------- #


def merge_dirs(
    src_dirs: List[pathlib.Path], merged_dir: pathlib.Path
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Merge Kaldi dirs.  Returns dictionaries of merged wav.scp and utt2spk
    (needed later).  If duplicate utterance IDs are found, the *first*
    occurrence wins and a warning is printed.
    """
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True)

    merged: Dict[str, Dict[str, str]] = {
        fname: collections.OrderedDict() for fname in KALDI_FILES_PRIORITY
    }

    for src in src_dirs:
        for fname in KALDI_FILES_PRIORITY:
            f = src / fname
            if not f.is_file():
                continue
            current = read_kaldi_kv(f)
            tgt = merged[fname]
            for k, v in current.items():
                if k in tgt and tgt[k] != v:
                    print(
                        f"[WARN] {fname}: duplicate key '{k}' – keeping first value.",
                        file=sys.stderr,
                    )
                    continue
                tgt[k] = v

    # write merged files
    for fname, kv in merged.items():
        if kv:  # skip empty
            write_kaldi_kv(kv, merged_dir / fname)

    # regenerate spk2utt
    if merged["utt2spk"]:
        write_kaldi_kv(make_spk2utt(merged["utt2spk"]), merged_dir / "spk2utt")

    return merged["wav.scp"], merged["utt2spk"]


# --------------------------------------------------------------------- #
#                    CV-fold generation
# --------------------------------------------------------------------- #


def split_speakers(
    utt2spk: Dict[str, str], seed: int = 2025
) -> Tuple[List[str], List[List[str]]]:
    """Return (special_spks, folds) where folds is a list of 5 speaker lists."""
    spk_set: Set[str] = set(utt2spk.values())
    special = [s for s in spk_set if any(t in s for t in SPECIAL_SUBSTRS)]
    ordinary = sorted(spk_set - set(special))

    random.Random(seed).shuffle(ordinary)
    # split into 5 roughly equal folds
    k = 5
    folds = [ordinary[i::k] for i in range(k)]
    return special, folds


def subset_kaldi(
    kaldi_dir: pathlib.Path, out_dir: pathlib.Path, speakers: Set[str]
) -> None:
    """
    Create a Kaldi data dir containing only utterances whose speaker
    is in `speakers`.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    utt2spk_path = kaldi_dir / "utt2spk"
    utt2spk_all = read_kaldi_kv(utt2spk_path)

    subset_utts = {u: s for u, s in utt2spk_all.items() if s in speakers}

    # Copy/trim all Kaldi two-column files present
    for fname in KALDI_FILES_PRIORITY:
        fsrc = kaldi_dir / fname
        if not fsrc.is_file():
            continue
        data = read_kaldi_kv(fsrc)
        trimmed = {
            k: v
            for k, v in data.items()
            if (k in subset_utts) or
            # For spk2utt files keys are speakers, not utts
            (fname.startswith("spk2") and k in speakers)
        }
        if trimmed:
            write_kaldi_kv(trimmed, out_dir / fname)

    # Write utt2spk and spk2utt (guaranteed to exist)
    write_kaldi_kv(subset_utts, out_dir / "utt2spk")
    write_kaldi_kv(make_spk2utt(subset_utts), out_dir / "spk2utt")


def make_folds(
    merged_dir: pathlib.Path,
    out_root: pathlib.Path,
    utt2spk: Dict[str, str],
    seed: int = 2025,
) -> None:
    special_spks, folds = split_speakers(utt2spk, seed)

    for k, val_spks in enumerate(folds, start=1):
        train_spks = set(special_spks) | (set().union(*folds) - set(val_spks))
        val_spks = set(val_spks)

        fold_dir = out_root / f"fold{k}"
        subset_kaldi(merged_dir, fold_dir / "train", train_spks)
        subset_kaldi(merged_dir, fold_dir / "val", val_spks)
        print(
            f"[INFO] Fold{k}: "
            f"{len(train_spks)} train speakers, {len(val_spks)} val speakers"
        )


# --------------------------------------------------------------------- #
#                    CLI
# --------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Merge Kaldi dirs and build 5-fold speaker CV splits "
        "with INV/SPEAKER speakers kept in train."
    )
    ap.add_argument(
        "src_dirs", nargs="+", type=pathlib.Path, help="Input Kaldi data dirs."
    )
    ap.add_argument(
        "--out-root",
        required=True,
        type=pathlib.Path,
        help="Directory where merged data and folds are written.",
    )
    ap.add_argument(
        "--seed", type=int, default=2025, help="Random seed (default 2025)."
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    merged_dir = args.out_root / "merged"
    print("[INFO] Merging source dirs …")
    _, utt2spk = merge_dirs(args.src_dirs, merged_dir)

    print("[INFO] Creating 5-fold speaker splits …")
    make_folds(merged_dir, args.out_root, utt2spk, args.seed)
    print("[DONE] All folds written under:", args.out_root.resolve())


if __name__ == "__main__":
    main()
