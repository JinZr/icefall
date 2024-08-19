#!/usr/bin/env python3
# Copyright      2024  Xiaomi Corp.        (authors: Xiaoyu Yang)
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
This script loads a checkpoint and uses it to decode waves.
You can generate the checkpoint with the following command:

Note: This is an example for the AudioSet dataset, if you are using different
dataset, you should change the argument values according to your dataset.

Usage of this script:

  repo_url=https://huggingface.co/marcoyang/icefall-audio-tagging-audioset-zipformer-2024-03-12
  repo=$(basename $repo_url)
  GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
  pushd $repo/exp
  git lfs pull --include pretrained.pt
  popd

  python3 zipformer/pretrained.py \
    --checkpoint $repo/exp/pretrained.pt \
    --label-dict $repo/data/class_labels_indices.csv \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/2.wav \
    $repo/test_wavs/3.wav \
    $repo/test_wavs/4.wav
"""


import argparse
import csv
import logging
import math
from itertools import groupby
from typing import List, Tuple

import kaldifeat
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from train import add_model_arguments, get_model, get_params


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint. "
        "The checkpoint is assumed to be saved by "
        "icefall.checkpoint.save_checkpoint().",
    )

    parser.add_argument(
        "--label-dict",
        type=str,
        help="""class_labels_indices.csv.""",
    )

    parser.add_argument(
        "sound_files",
        type=str,
        nargs="+",
        help="The input sound file(s) to transcribe. "
        "Supported formats are those supported by torchaudio.load(). "
        "For example, wav and flac are supported. "
        "The sample rate has to be 16kHz.",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="The sample rate of the input sound file",
    )

    parser.add_argument(
        "--audio-chunk-size",
        type=int,
        default=20,
        help="The duration of each chunk (in second).",
    )

    parser.add_argument(
        "--nc",
        type=int,
        default=20,
        help="The number of batch-fy chunks.",
    )

    add_model_arguments(parser)

    return parser


def read_sound_files(
    filenames: List[str], expected_sample_rate: float
) -> Tuple[List[torch.Tensor], List[float]]:
    """Read a list of sound files into a list 1-D float32 torch tensors.
    Args:
      filenames:
        A list of sound filenames.
      expected_sample_rate:
        The expected sample rate of the sound files.
    Returns:
      Return a list of 1-D float32 torch tensors.
    """
    ans = []
    dur = []
    assert len(filenames) == 1, "Only one sound file is supported"
    for f in filenames:
        wave, sample_rate = torchaudio.load(f)
        assert (
            sample_rate == expected_sample_rate
        ), f"expected sample rate: {expected_sample_rate}. Given: {sample_rate}"
        # We use only the first channel
        ans.append(wave[0].contiguous())
        dur.append(wave.size(-1) / sample_rate / 60 / 60)
    return ans, dur


def merge_adjascent_chunks(arr: List[int]):
    merged = []
    for key, group in groupby(arr, lambda x: x):
        if list(group)[0] == 1:
            merged.append(1)
        else:
            merged += list(group)
    return merged


def read_n_chunks(
    wave: torch.Tensor, sample_rate: int, audio_chunk_size: int
) -> List[torch.Tensor]:
    """Read a list of sound files into a list 1-D float32 torch tensors.
    Args:
      wave:
        torch.Tensor
      sample_rate:
        The expected sample rate of the sound file.
      audio_chunk_size:
        The duration of each chunk (in second).
      nc:
        The number of batch-fy chunks.
    Returns:
      Return a list of 1-D float32 torch tensors.
    """
    ans = []
    wave_len = wave.size(-1)
    print(wave.size())
    chunk_len = sample_rate * audio_chunk_size
    nc = wave_len // chunk_len
    for i in range(nc):
        chunk = wave[
            (i * chunk_len) : min((i + 1) * chunk_len, wave_len),
        ]
        ans.append(chunk)
    return ans


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()

    params = get_params()

    params.update(vars(args))

    logging.info(f"{params}")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    logging.info("Creating model")
    model = get_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    model.eval()

    # get the label dictionary
    label_dict = {}
    with open(params.label_dict, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, row in enumerate(reader):
            if i == 0:
                continue
            label_dict[int(row[0])] = row[2]

    logging.info("Constructing Fbank computer")
    opts = kaldifeat.FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = params.sample_rate
    opts.mel_opts.num_bins = params.feature_dim
    opts.mel_opts.high_freq = -400

    fbank = kaldifeat.Fbank(opts)

    logging.info(f"Reading sound files: {params.sound_files}")
    waves, wave_durs = read_sound_files(
        filenames=params.sound_files, expected_sample_rate=params.sample_rate
    )
    wave_lens = [w.size(-1) for w in waves]

    audio_chunk_size = params.audio_chunk_size
    logging.info(f"Chunk size: {audio_chunk_size}")

    logging.info("Decoding started")
    wave_labels = []
    for wave_index, _ in enumerate(wave_lens):
        chunks = read_n_chunks(waves[wave_index], params.sample_rate, audio_chunk_size)
        wave_label = []
        for chunk_index in range(0, len(chunks), params.nc):
            features = fbank(chunks[chunk_index : chunk_index + params.nc])
            features = [f.to(device) for f in features]
            feature_lengths = [f.size(0) for f in features]

            features = pad_sequence(
                features, batch_first=True, padding_value=math.log(1e-10)
            )
            feature_lengths = torch.tensor(feature_lengths, device=device)

            # model forward and predict the audio events
            encoder_out, encoder_out_lens = model.forward_encoder(
                features, feature_lengths
            )
            logits = model.forward_audio_tagging(encoder_out, encoder_out_lens)

            for filename, logit in zip(args.sound_files, logits):
                topk_prob, topk_index = logit.sigmoid().topk(5)
                # topk_labels = [label_dict[index.item()] for index in topk_index]
                topk_labels = [int(index.item() == 43) for index in topk_index]
                wave_label += topk_labels
        wave_labels.append(wave_label)

    logging.info("Done")
    for i, (wave_label, wave_dur) in enumerate(zip(wave_labels, wave_durs)):
        print(f"Wave {i}: {wave_label} \n")
        print(f"``Snoring`` detected in {sum(wave_label)} chunks")
        print(f"Duration: {wave_dur} hours")
        print("\n")
        print(f"Estimated AHI index: {sum(wave_label) / wave_dur}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
