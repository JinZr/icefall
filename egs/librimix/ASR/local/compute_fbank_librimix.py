import logging
import os
from pathlib import Path

import torch
import torch.multiprocessing
from lhotse import (
    CutSet,
    Fbank,
    FbankConfig,
    LilcomChunkyWriter,
    SupervisionSegment,
    load_manifest,
)
from lhotse.cut import CutSet, MonoCut
from lhotse.recipes.utils import read_manifests_if_cached
from tqdm.auto import tqdm

from icefall.utils import get_executor, str2bool

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.multiprocessing.set_sharing_strategy("file_system")


def compute_fbank_librimix(n_src: int, part: str):
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")

    sampling_rate = 16000
    num_mel_bins = 80
    num_jobs = min(15, os.cpu_count())
    num_mel_bins = 80

    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    with get_executor() as ex:  # Initialize the executor only once.
        supervision_set = src_dir.joinpath(f"librispeech_supervisions_{part}.jsonl.gz")

        logging.info("Reading manifests")
        if not os.path.exists(supervision_set):
            raise FileNotFoundError(f"Manifest {supervision_set} not found")
        supervision_set = load_manifest(supervision_set)

        logging.info("Reading cuts")
        recording_set = load_manifest(
            src_dir.joinpath(f"librimix_{n_src}mix_{part}_recordings_mix_both.jsonl.gz")
        )
        cuts = []
        for recording in tqdm(recording_set):
            recording_ids = recording.recording_id.split("_")
            cuts.append(
                MonoCut(
                    id=recording.recording_id,
                    start=0,
                    duration=recording.duration,
                    channel=recording.channel,
                    recording=recording,
                    channel=0,
                    supervisions=merge_supervision_segments(
                        recording_id=recording.recording_id,
                        duration=recording.duration,
                        supervisions=[
                            supervision_set.find(
                                recording_id=recording_id,
                            )
                            for recording_id in recording_ids
                        ],
                    ),
                )
            )
        cuts = CutSet.from_cuts(cuts)
        cuts.compute_and_store_features(
            extractor=extractor,
            storage_path=f"{output_dir}/librimix_{n_src}mix_feats_{part}",
            # when an executor is specified, make more partitions
            num_jobs=num_jobs if ex is None else 80,
            executor=ex,
            storage_type=LilcomChunkyWriter,
        )


def merge_supervision_segments(recording_id, duration, supervisions):
    return [
        SupervisionSegment(
            id=recording_id,
            recording_id=recording_id,
            start=0.0,
            duration=duration,
            speaker="_".join([s.speaker for s in supervisions]),
            text="<sc>".join([s.text for s in supervisions]),
        )
    ]


if __name__ == "__main__":
    for part in ["train-100", "train-360"]:
        compute_fbank_librimix(n_src=2, part=part)
