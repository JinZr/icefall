#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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
This file computes fbank features of the LibriSpeech dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""

import argparse
import logging
import os
from pathlib import Path

import torch
from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import get_executor, str2bool

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--perturb-speed",
        type=str2bool,
        default=True,
        help="""Perturb speed with factor 0.9 and 1.1 on train subset.""",
    )
    return parser.parse_args()


def compute_fbank_osa(args):
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")
    num_jobs = min(15, os.cpu_count())
    num_mel_bins = 80

    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    dataset_parts = ("all",)

    prefix = "osa"
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

    with get_executor() as ex:  # Initialize the executor only once.
        for partition, m in manifests.items():
            cuts_filename = f"{prefix}_cuts_{partition}.{suffix}"
            if (output_dir / cuts_filename).is_file():
                logging.info(f"{partition} already exists - skipping.")
                continue
            logging.info(f"Processing {partition}")
            cut_set = CutSet.from_manifests(
                recordings=m["recordings"],
                supervisions=m["supervisions"],
            )

            if "train" in partition:
                if args.perturb_speed:
                    logging.info(f"Doing speed perturb")
                    cut_set = (
                        cut_set
                        + cut_set.perturb_speed(0.9)
                        + cut_set.perturb_speed(1.1)
                    )
            cut_set = cut_set.compute_and_store_features(
                extractor=extractor,
                storage_path=f"{output_dir}/{prefix}_feats_{partition}",
                # when an executor is specified, make more partitions
                num_jobs=num_jobs if ex is None else 80,
                executor=ex,
                storage_type=LilcomChunkyWriter,
            )
            # logging.info("About to split cuts into smaller chunks.")
            # cut_set = cut_set.trim_to_supervisions(
            #     keep_overlapping=False, min_duration=None
            # )

            logging.info(f"Saving to {cuts_filename}")
            cut_set.to_file(output_dir / cuts_filename)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    logging.info(vars(args))
    compute_fbank_osa(args)
