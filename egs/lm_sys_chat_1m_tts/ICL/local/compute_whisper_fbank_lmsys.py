#!/usr/bin/env python3
# Copyright    2024  Xiaomi Corp.        (authors: Zengrui Jin)
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
This file computes fbank features of the aishell dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""

import argparse
import logging
import os
from pathlib import Path

import torch
from lhotse import CutSet, LilcomChunkyWriter, WhisperFbank, WhisperFbankConfig
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import get_executor, str2bool

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def compute_fbank_aishell(
    num_mel_bins: int = 128,
    output_dir: str = "data/fbank_whisper",
):
    src_dir = Path("data/manifests")
    output_dir = Path(output_dir)
    num_jobs = min(15, os.cpu_count())

    dataset_parts = "train"
    prefix = "lmsys"
    suffix = "jsonl.gz"
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix=prefix,
        suffix=suffix,
    )
    assert manifests is not None

    assert len(manifests) == len(dataset_parts), (
        len(manifests),
        len(dataset_parts),
        list(manifests.keys()),
        dataset_parts,
    )
    extractor = WhisperFbank(
        WhisperFbankConfig(num_filters=num_mel_bins, device="cuda")
    )

    with get_executor() as ex:  # Initialize the executor only once.
        for partition, m in manifests.items():
            if (output_dir / f"{prefix}_cuts_{partition}.{suffix}").is_file():
                logging.info(f"{partition} already exists - skipping.")
                continue
            logging.info(f"Processing {partition}")
            cut_set = CutSet.from_manifests(
                recordings=m["recordings"],
                supervisions=m["supervisions"],
            )

            cut_set = cut_set.compute_and_store_features(
                extractor=extractor,
                storage_path=f"{output_dir}/{prefix}_feats_{partition}",
                # when an executor is specified, make more partitions
                num_jobs=num_jobs if ex is None else 80,
                executor=ex,
                storage_type=LilcomChunkyWriter,
            )
            cut_set.to_file(output_dir / f"{prefix}_cuts_{partition}.{suffix}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-mel-bins",
        type=int,
        default=128,
        help="""The number of mel bins for Whisper Fbank""",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/fbank_whisper",
        help="Output directory. Default: data/fbank_whisper.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()
    compute_fbank_aishell(
        num_mel_bins=args.num_mel_bins,
        output_dir=args.output_dir,
    )
