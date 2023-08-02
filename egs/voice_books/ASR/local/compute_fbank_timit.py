#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang
#                                                  Mingshuang Luo)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
This file computes fbank features of the TIMIT dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""

import logging
import os
from pathlib import Path

import torch
from lhotse import (
    CutSet,
    Fbank,
    FbankConfig,
    LilcomChunkyWriter,
    MonoCut,
    SupervisionSegment,
    KaldifeatFbank,
    KaldifeatFbankConfig,
)
from lhotse.recipes.utils import (
    read_manifests_if_cached,
)
from lhotse import set_audio_duration_mismatch_tolerance, set_caching_enabled
from icefall.utils import get_executor

from tqdm import tqdm

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def compute_fbank_timit():
    src_dir = Path("data/manifests")
    num_jobs = min(15, os.cpu_count())
    num_mel_bins = 80

    subset = "part1"
    prefix = "openspeech"
    suffix = "jsonl.gz"

    num_splits = 50
    start = 0
    stop = num_splits
    output_dir = f"data/fbank/split_{num_splits}"
    num_digits = len(str(num_splits))
    output_dir = Path(output_dir)
    assert output_dir.exists(), f"{output_dir} does not exist!"

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    extractor = KaldifeatFbank(KaldifeatFbankConfig(device=device))
    logging.info(f"device: {device}")

    set_audio_duration_mismatch_tolerance(0.01)  # 10ms tolerance
    set_caching_enabled(False)
    for i in range(start, stop):
        idx = f"{i + 1}".zfill(num_digits)
        logging.info(f"Processing {idx}/{num_splits}")

        cuts_path = output_dir / f"{prefix}_cuts_{subset}.{idx}.jsonl.gz"
        if cuts_path.is_file():
            logging.info(f"{cuts_path} exists - skipping")
            continue

        cutset = CutSet.from_file(
            src_dir / f"split_{num_splits}" / f"openspeech_cuts_part1.{idx}.jsonl.gz"
        )
        assert cutset is not None

        cut_set = CutSet.from_cuts(
            [
                MonoCut(
                    id=cut.id,
                    start=cut.start,
                    duration=cut.duration,
                    channel=cut.channel,
                    supervisions=[
                        SupervisionSegment(
                            id=cut.id,
                            recording_id=cut.supervisions[0].recording_id,
                            start=cut.supervisions[0].start,
                            duration=cut.supervisions[0].duration,
                            channel=cut.supervisions[0].channel,
                            text=cut.supervisions[0].custom["texts"][0],
                        )
                    ],
                    recording=cut.recording,
                )
                for cut in tqdm(cutset)
            ]
        )

        logging.info("Splitting cuts into smaller chunks.")
        cut_set = cut_set.trim_to_supervisions(
            keep_overlapping=False, min_duration=None
        )

        cut_set = cut_set.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=f"{output_dir}/{prefix}_feats_{subset}",
            # when an executor is specified, make more partitions
            num_workers=20,
            storage_type=LilcomChunkyWriter,
            batch_duration=1200,
        )
        cut_set.to_file(cuts_path)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    compute_fbank_timit()
