# Copyright      2021  Piotr Żelasko
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


import argparse
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from lhotse import CutSet, RecordingSet, load_manifest_lazy
from lhotse.dataset import CutConcatenate, DynamicBucketingSampler, SimpleCutSampler
from lhotse.utils import fix_random_seed
from librimix_sep_dataset import LibriMixSpeechSeparationDataset
from torch.utils.data import DataLoader

from icefall.utils import str2bool


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class LibriMixSpeechSeparationDataModule:
    """
    DataModule for k2 ASR experiments.
    It assumes there is always one train and valid dataloader,
    but there can be multiple test dataloaders (e.g. LibriSpeech test-clean
    and test-other).

    It contains all the common data pipeline modules used in ASR
    experiments, e.g.:
    - dynamic batch size,
    - bucketing samplers,
    - cut concatenation,
    - augmentation,

    This class should be derived for specific corpora used in ASR tasks.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="Speech separation data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
            "effective batch sizes, sampling strategies, applied data "
            "augmentations, etc.",
        )
        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("data/manifests"),
            help="Path to directory with train/valid/test cuts.",
        )
        group.add_argument(
            "--max-duration",
            type=int,
            default=200.0,
            help="Maximum pooled recordings duration (seconds) in a "
            "single batch. You can reduce it if it causes CUDA OOM.",
        )
        group.add_argument(
            "--chunk-duration",
            type=int,
            default=4.0,
        )
        group.add_argument(
            "--max-duration",
            type=int,
            default=200.0,
            help="Maximum pooled recordings duration (seconds) in a "
            "single batch. You can reduce it if it causes CUDA OOM.",
        )
        group.add_argument(
            "--bucketing-sampler",
            type=str2bool,
            default=True,
            help="When enabled, the batches will come from buckets of "
            "similar duration (saves padding frames).",
        )
        group.add_argument(
            "--num-buckets",
            type=int,
            default=30,
            help="The number of buckets for the DynamicBucketingSampler"
            "(you might want to increase it for larger datasets).",
        )
        group.add_argument(
            "--concatenate-cuts",
            type=str2bool,
            default=False,
            help="When enabled, utterances (cuts) will be concatenated "
            "to minimize the amount of padding.",
        )
        group.add_argument(
            "--duration-factor",
            type=float,
            default=1.0,
            help="Determines the maximum duration of a concatenated cut "
            "relative to the duration of the longest cut in a batch.",
        )
        group.add_argument(
            "--gap",
            type=float,
            default=1.0,
            help="The amount of padding (in seconds) inserted between "
            "concatenated cuts. This padding is filled with noise when "
            "noise augmentation is used.",
        )
        group.add_argument(
            "--shuffle",
            type=str2bool,
            default=True,
            help="When enabled (=default), the examples will be "
            "shuffled for each epoch.",
        )
        group.add_argument(
            "--return-cuts",
            type=str2bool,
            default=True,
            help="When enabled, each batch will have the "
            "field: batch['supervisions']['cut'] with the cuts that "
            "were used to construct it.",
        )
        group.add_argument(
            "--num-workers",
            type=int,
            default=8,
            help="The number of training dataloader workers that "
            "collect the batches.",
        )

    def train_dataloaders(
        self,
        source_cuts: CutSet,
        mixture_cuts: CutSet,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
    ) -> DataLoader:
        """
        Args:
          cuts_train:
            CutSet for training.
          sampler_state_dict:
            The state dict for the training sampler.
        """

        logging.info("About to create train dataset")
        train = LibriMixSpeechSeparationDataset(
            sources_set=source_cuts,
            mixtures_set=mixture_cuts,
            chunk_duration=self.args.chunk_duration,
        )

        if self.args.bucketing_sampler:
            logging.info("Using DynamicBucketingSampler.")
            train_sampler = DynamicBucketingSampler(
                mixture_cuts,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                drop_last=self.args.drop_last,
            )
        else:
            logging.info("Using SimpleCutSampler.")
            train_sampler = SimpleCutSampler(
                mixture_cuts,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
            )

        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            train_sampler.load_state_dict(sampler_state_dict)

        logging.info("About to create train dataloader")

        # 'seed' is derived from the current random state, which will have
        # previously been set in the main process.
        seed = torch.randint(0, 100000, ()).item()
        worker_init_fn = _SeedWorkers(seed)

        train_dl = DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
            worker_init_fn=worker_init_fn,
        )

        return train_dl

    @lru_cache()
    def train_100_source_cuts(self) -> CutSet:
        logging.info("About to get train-100 source cuts")
        recordings = RecordingSet.from_jsonl(
            self.args.manifest_dir
            / "librimix_2mix_train-100_recordings_sources_both.jsonl.gz"
        )
        return CutSet.from_manifests(recordings=recordings)

    @lru_cache()
    def train_100_mixture_cuts(self) -> CutSet:
        logging.info("About to get train-100 mixture cuts")
        recordings = RecordingSet.from_jsonl(
            self.args.manifest_dir
            / "librimix_2mix_train-100_recordings_mix_both.jsonl.gz"
        )
        return CutSet.from_manifests(recordings=recordings)

    @lru_cache()
    def train_360_source_cuts(self) -> CutSet:
        logging.info("About to get train-360 source cuts")
        recordings = RecordingSet.from_jsonl(
            self.args.manifest_dir
            / "librimix_2mix_train-360_recordings_sources_both.jsonl.gz"
        )
        return CutSet.from_manifests(recordings=recordings)

    @lru_cache()
    def train_360_mixture_cuts(self) -> CutSet:
        logging.info("About to get train-360 mixture cuts")
        recordings = RecordingSet.from_jsonl(
            self.args.manifest_dir
            / "librimix_2mix_train-360_recordings_mix_both.jsonl.gz"
        )
        return CutSet.from_manifests(recordings=recordings)

    @lru_cache()
    def test_source_cuts(self) -> CutSet:
        logging.info("About to get test source cuts")
        recordings = RecordingSet.from_jsonl(
            self.args.manifest_dir
            / "librimix_2mix_test_recordings_sources_both.jsonl.gz"
        )
        return CutSet.from_manifests(recordings=recordings)

    @lru_cache()
    def test_mixture_cuts(self) -> CutSet:
        logging.info("About to get test mixture cuts")
        recordings = RecordingSet.from_jsonl(
            self.args.manifest_dir / "librimix_2mix_test_recordings_mix_both.jsonl.gz"
        )
        return CutSet.from_manifests(recordings=recordings)

    @lru_cache()
    def dev_source_cuts(self) -> CutSet:
        logging.info("About to get dev source cuts")
        recordings = RecordingSet.from_jsonl(
            self.args.manifest_dir
            / "librimix_2mix_dev_recordings_sources_both.jsonl.gz"
        )
        return CutSet.from_manifests(recordings=recordings)

    @lru_cache()
    def dev_mixture_cuts(self) -> CutSet:
        logging.info("About to get dev mixture cuts")
        recordings = RecordingSet.from_jsonl(
            self.args.manifest_dir / "librimix_2mix_dev_recordings_mix_both.jsonl.gz"
        )
        return CutSet.from_manifests(recordings=recordings)
