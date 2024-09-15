import argparse

from lhotse import CutSet, SupervisionSegment, load_manifest_lazy
from tqdm import tqdm

input_cuts = load_manifest_lazy("data/fbank_5s/train_recorder.jsonl.gz")

for cut in tqdm(input_cuts, desc="Converting supervisions"):
    try:
        print(cut.supervisions[0].audio_event)
    except:
        print(cut)
        exit()
