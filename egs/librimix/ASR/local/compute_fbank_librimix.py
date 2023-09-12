import logging
import os
import random
import warnings
from pathlib import Path

import torch
import torch.multiprocessing
from lhotse import LilcomChunkyWriter, load_manifest
from lhotse.cut import CutSet
from lhotse.features.kaldifeat import (
    KaldifeatFbank,
    KaldifeatFbankConfig,
    KaldifeatFrameOptions,
    KaldifeatMelOptions,
)
from lhotse.recipes.utils import read_manifests_if_cached

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.multiprocessing.set_sharing_strategy("file_system")


def compute_fbank_lsmix():
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")

    sampling_rate = 16000
    num_mel_bins = 80

    extractor = KaldifeatFbank(
        KaldifeatFbankConfig(
            frame_opts=KaldifeatFrameOptions(sampling_rate=sampling_rate),
            mel_opts=KaldifeatMelOptions(num_bins=num_mel_bins),
            device="cuda",
        )
    )

    clean100 = src_dir + "librispeech_supervisions_train-clean-100.jsonl.gz"
    clean360 = src_dir + "librispeech_supervisions_train-clean-360.jsonl.gz"

    logging.info("Reading manifests")
    if not os.path.exists(clean100):
        raise FileNotFoundError(f"Manifest {clean100} not found")
    if not os.path.exists(clean360):
        raise FileNotFoundError(f"Manifest {clean360} not found")
    supervisions = load_manifest(clean100) + load_manifest(clean360)
