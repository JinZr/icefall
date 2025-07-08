import json
from collections import defaultdict
from pathlib import Path


def gather_by_index(json_path: str):
    """
    Build {index: {utt_id: transcript}} for every entry in the JSON file.

    • Variant suffixes like “-2” stay on the utt_id so you don’t lose track
      of duplicates.
    • The index is taken from everything after the last “_” in the (base) ID,
      e.g.  CN_HN_061  →  “061”.
    """
    data = json.loads(Path(json_path).read_text())
    by_index = defaultdict(dict)

    for utt_id, text_list in data.items():
        transcript = text_list[0] if isinstance(text_list, list) else text_list
        core_id = utt_id.split('-')[0]        # strip “-2”, “-3”, etc.
        index = core_id.split('_')[-1]        # grab the numeric tail
        by_index[index][utt_id] = transcript  # keep full utt_id

    return dict(by_index)

if __name__ == "__main__":
    all_indices = gather_by_index("./local/hyper_text.json")

    # quick demo – print everything, grouped and sorted numerically
    for idx, utt_dict in sorted(all_indices.items(), key=lambda x: int(x[0])):
        print(f"Index {idx}:")
        for utt_id, transcript in utt_dict.items():
            print(f"  {utt_id}: {transcript}")
        print()