import argparse
import random
from pathlib import Path

import lhotse
from lhotse import CutSet
from tqdm.auto import tqdm

MAPPING = {
    0:0,3:0,5:0,1:1,2:1,4:1
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Path to the input manifest', required=True)
    parser.add_argument('--output', type=str, help='Path to the output manifest', required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    input_manifest = Path(args.input)
    output_manifest = Path(args.output)

    cut_set = lhotse.load_manifest(args.input)
    output_cut_set = []
    normal_cut_set = []
    total, normal, abnormal = 0, 0, 0
    
    for cut in tqdm(cut_set):
        cut_id = cut.id
        sup = cut.supervisions[0]
        total += 1
        for label in sup.text.split(";"):
            if MAPPING[int(label)] == 1:
                abnormal += 1
                output_cut_set.append(cut)
                continue
        normal += 1
        normal_cut_set.append(cut)
    print(f"Total: {total}, Normal: {normal}, Abnormal: {abnormal}")
    output_cut_set += random.choices(normal_cut_set, k=abnormal)
    random.shuffle(output_cut_set)
    CutSet.from_cuts(output_cut_set).to_jsonl(output_manifest)
    # lhotse.save_manifest(output_cut_set, output_manifest)



    