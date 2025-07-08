#!/usr/bin/env python3
"""
Generate training index (train.csv) for the CLP database.

Output CSV columns (fixed, no option switch):

    utt_id, spk_id, path, text

* Recursively scan `data_root` (default `data/clp_database`)
* `control` / `hyper` transcripts are read from `control_text.json` / `hyper_text.json`
* For `control_text`, if the full utterance‑id is missing fall back to its numeric suffix
* Text normalization: remove punctuation, collapse spaces, **convert to UPPER CASE**
* Speaker‑id is the substring before the first underscore in the utterance‑id (e.g. `EC1_NH_023` → `EC1`)
"""
import argparse
import csv
import json
import os
import re
import unicodedata
from pathlib import Path

###############################################################################
# --------------------------- text normalisation -----------------------------
###############################################################################
_PUNCT = re.compile(r"[^A-Za-z0-9' ]+")
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    """ASR-centred text normalisation."""
    text = unicodedata.normalize("NFKD", text)  # 兼容有重音/全角字符等情况
    text = _PUNCT.sub(" ", text)  # 去掉除 a-z 0-9 和 ' 之外的符号
    text = _SPACES.sub(" ", text).strip()
    text = text.upper()
    return text


###############################################################################
# ------------------------- helper for control part --------------------------
###############################################################################
_DIGIT_SUFFIX = re.compile(r"(\d+(?:[-_]\d+)?)$")  # 例如 065-2  / 060_1  / 060


def get_control_trn(utt_id: str, ctrl_dict: dict) -> str | None:
    if utt_id in ctrl_dict:
        return ctrl_dict[utt_id][0]

    m = _DIGIT_SUFFIX.search(utt_id)
    if m:
        key = m.group(1).lstrip("_")  # 去掉可能的前置 _
        if key in ctrl_dict:
            return ctrl_dict[key][0]
    return None


###############################################################################
# ------------------------------ main routine ---------------------------------
###############################################################################
def main(args):
    # 读取 json 标注
    with open(args.control_json, encoding="utf-8") as f:
        control_trn = json.load(f)
    with open(args.hyper_json, encoding="utf-8") as f:
        hyper_trn = json.load(f)

    rows = []
    for root, _, files in os.walk(args.data_root):
        for fn in files:
            if not fn.lower().endswith(".wav"):
                continue
            full_path = os.path.join(root, fn)
            # rel_path = os.path.relpath(full_path, args.data_root)
            utt_id = os.path.splitext(fn)[0]
            spk_id = utt_id.split("_")[0]

            if "/control/" in full_path.replace("\\", "/"):
                text = get_control_trn(utt_id, control_trn)
            else:
                text = hyper_trn.get(utt_id, [None])[0] if utt_id in hyper_trn else None

            if text is None:
                continue

            norm_text = normalize(text)
            rows.append([utt_id, spk_id, full_path, norm_text])

    rows.sort()  # 可让 csv 行固定（按路径排序）

    header = ["utt_id", "spk_id", "path", "text"]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # writer.writerow(header) 
        writer.writerows(rows)

    print(f"✔ Wrote {len(rows)} lines to {args.output_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-root",
        default="data/clp_database",
        help="dataset root (contains control/ & hyper/)",
    )
    p.add_argument("--control-json", default="./local/control_text.json")
    p.add_argument("--hyper-json", default="./local/hyper_text.json")
    p.add_argument(
        "--output-csv", type=Path, required=True, help="output CSV file path"
    )
    main(p.parse_args())
