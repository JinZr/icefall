#!/usr/bin/env python3
"""
generate_kfold.py

Create k-fold (default = 5) splits for ASR data, keeping “NH” utterances always in training and performing **speaker‑based** cross‑validation over “HN” utterances.

Usage
-----
python generate_kfold.py data.csv  --out_dir data/5-folds  --k 5
"""
import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold


def write_kaldi_files(df: pd.DataFrame, dst: Path) -> None:
    """Write Kaldi-style files (wav.scp, text, utt2spk, spk2utt)."""
    dst.mkdir(parents=True, exist_ok=True)

    # wav.scp
    (dst / "wav.scp").write_text("\n".join(
        f"{u} {p}" for u, p in zip(df.utt_id, df.path)
    ))

    # text
    (dst / "text").write_text("\n".join(
        f"{u} {t}" for u, t in zip(df.utt_id, df.transcript)
    ))

    # utt2spk
    (dst / "utt2spk").write_text("\n".join(
        f"{u} {s}" for u, s in zip(df.utt_id, df.spk_id)
    ))

    # spk2utt (inverse map)
    spk2utt_lines = []
    for spk, g in df.groupby("spk_id"):
        utts = " ".join(g.utt_id)
        spk2utt_lines.append(f"{spk} {utts}")
    (dst / "spk2utt").write_text("\n".join(spk2utt_lines))


def main(args):
    df = pd.read_csv(args.csv)

    # Basic sanity check
    required_cols = {"utt_id", "spk_id", "path", "transcript"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns in CSV: {missing}")

    df_nh = df[df.utt_id.str.contains("NH")]
    df_hn = df[df.utt_id.str.contains("HN")]
    # Shuffle once to randomize speaker order while keeping reproducibility
    df_hn = df_hn.sample(frac=1, random_state=1234).reset_index(drop=True)

    gkf = GroupKFold(n_splits=args.k)

    out_root = Path(args.out_dir)
    for fold_idx, (train_idx, val_idx) in enumerate(
            gkf.split(df_hn, groups=df_hn.spk_id), start=1):
        fold_dir = out_root / f"fold{fold_idx}"
        train_dir = fold_dir / "train"
        val_dir = fold_dir / "val"

        # Assemble splits
        df_val = df_hn.iloc[val_idx].reset_index(drop=True)
        df_train = pd.concat([df_nh, df_hn.iloc[train_idx]], ignore_index=True)

        # Write CSVs
        fold_dir.mkdir(parents=True, exist_ok=True)
        df_train.to_csv(fold_dir / "train.csv", index=False, header=False)
        df_val.to_csv(fold_dir / "val.csv", index=False, header=False)

        # Write Kaldi files
        write_kaldi_files(df_train, train_dir)
        write_kaldi_files(df_val, val_dir)

        print(f"[Fold {fold_idx}]  train={len(df_train):5d}  val={len(df_val):5d}")

    print(f"\nDone. Splits written to {out_root.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="k-fold CV with NH locked in train")
    parser.add_argument("csv", help="Input data.csv (utt_id, spk_id, path, transcript)")
    parser.add_argument("--out_dir", default="data/clp-5-folds",
                        help="Output root directory (default: data/5-folds)")
    parser.add_argument("-k", "--k", type=int, default=5,
                        help="Number of folds (default: 5)")
    main(parser.parse_args())