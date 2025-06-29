import argparse
import json
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser(
        description="Dump units from Kaldi data directory to JSON."
    )
    parser.add_argument(
        "--cn-units", type=Path, required=True, help="Chinese units file"
    )
    parser.add_argument(
        "--en-units", type=Path, required=True, help="English units file"
    )
    parser.add_argument(
        "--out", type=Path, default="units.json", help="Output JSON file"
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    cn_units_lines = args.cn_units.read_text().strip().split("\n")
    cn_units = [line.split(" ")[0] for line in cn_units_lines if line]
    en_units_lines = args.en_units.read_text().strip().split("\n")
    en_units = [line.split(" ")[0] for line in en_units_lines if line]

    if not cn_units or not en_units:
        raise ValueError("Units files cannot be empty.")

    all_units = ["|"] + cn_units + en_units
    unit_dict = {v: k for k, v in enumerate(all_units)}
    unit_dict["[UNK]"] = len(unit_dict)
    unit_dict["[PAD]"] = len(unit_dict)

    with args.out.open("w", encoding="utf-8") as f:
        json.dump(unit_dict, f, ensure_ascii=False, indent=2)

    print(
        f"✓ Dumped {len(cn_units)} Chinese and {len(en_units)} English units to {args.out}"
    )


if __name__ == "__main__":
    main()
