#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Xiaoyu Yang)
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
Usage:

export CUDA_VISIBLE_DEVICES="0"

./zipformer/evaluate.py \
  --epoch 50 \
  --avg 10 \
  --exp-dir zipformer/exp \
  --max-duration 1000


"""

import argparse
import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from at_datamodule import OsaAtDatamodule
from lhotse import CutSet

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        recall_score,
    )
except Exception as ex:
    raise RuntimeError(f"{ex}\nPlease run\n" "pip3 install -U scikit-learn")
from train import add_model_arguments, get_model, get_params, str2onehot

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.utils import AttributeDict, setup_logger, str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=15,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="The threshold for audio tagging",
    )

    add_model_arguments(parser)

    return parser


def inference_one_batch(
    params: AttributeDict,
    model: nn.Module,
    batch: dict,
):
    device = next(model.parameters()).device
    feature = batch["inputs"]
    assert feature.ndim == 3, feature.shape

    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]
    audio_event = supervisions["audio_event"]

    label, _ = str2onehot(
        audio_event,
        n_classes=params.num_events,
        id_mapping={
            0: 0,
            3: 0,
            5: 0,
            1: 1,
            2: 1,
            4: 1,
        },
    )
    label = label.detach().cpu()

    feature_lens = supervisions["num_frames"].to(device)

    encoder_out, encoder_out_lens = model.forward_encoder(feature, feature_lens)

    audio_logits = model.forward_audio_tagging(encoder_out, encoder_out_lens)
    # convert to probabilities between 0-1
    audio_logits = audio_logits.sigmoid().detach().cpu()

    return audio_logits, label


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
) -> Dict:
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    all_logits = []
    all_labels = []

    for batch_idx, batch in enumerate(dl):
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]
        num_cuts += len(cut_ids)

        audio_logits, labels = inference_one_batch(
            params=params,
            model=model,
            batch=batch,
        )

        all_logits.append(audio_logits)
        all_labels.append(labels)

        if batch_idx % 20 == 1:
            logging.info(f"Processed {num_cuts} cuts already.")
    logging.info("Finish collecting audio logits")

    return all_logits, all_labels


@torch.no_grad()
def main():
    parser = get_parser()
    OsaAtDatamodule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    params.res_dir = params.exp_dir / "inference_audio_tagging"

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    if params.use_averaged_model:
        params.suffix += "-use-averaged-model"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Evaluation started")

    logging.info(params)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info("About to create model")

    model = get_model(params)

    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(
                average_checkpoints(filenames, device=device), strict=False
            )
        elif params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(
                average_checkpoints(filenames, device=device), strict=False
            )
    else:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg + 1
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(
                "Calculating the averaged model over iteration checkpoints"
                f" from {filename_start} (excluded) to {filename_end}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                ),
                strict=False,
            )
        else:
            assert params.avg > 0, params.avg
            start = params.epoch - params.avg
            assert start >= 1, start
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            logging.info(
                f"Calculating the averaged model over epoch range from "
                f"{start} (excluded) to {params.epoch}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                ),
                strict=False,
            )

    model.to(device)
    model.eval()

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    args.return_cuts = True
    osa = OsaAtDatamodule(args)

    osa_cuts = CutSet(cuts=[])
    if params.use_attached:
        osa_cuts = osa.osa_test_cuts()
    if params.use_recorder:
        osa_cuts += osa.osa_recorder_test_cuts()

    osa_dl = osa.test_dataloaders(osa_cuts)

    test_sets = ["osa_eval"]

    logits, labels = decode_dataset(
        dl=osa_dl,
        params=params,
        model=model,
    )

    logits = torch.cat(logits, dim=0).squeeze(dim=1).detach()
    labels = torch.cat(labels, dim=0).long().detach()
    acc_count = 0
    total_count = 0
    for logit, label in zip(torch.argmax(logits, dim=-1).numpy(), torch.argmax(labels, dim=-1).numpy()):
        total_count += 5
        acc_count += 5 - sum(abs(logit - label))

    # compute the metric
    # mAP = average_precision_score(
    #     y_true=labels,
    #     y_score=logits,
    # )

    # logging.info(f"mAP for audioset eval is: {mAP}")
    threshold = params.threshold # Zengrui Note: originally it was 0.6 
    acc = acc_count / total_count
    # acc = accuracy_score(labels, logits > threshold)
    # f1 = f1_score(labels, logits > threshold, average="micro")
    # recall = recall_score(labels, logits > threshold, average="micro")

    logging.info(f"Threshold for OSA eval is: {threshold}")
    logging.info(f"Accuracy for OSA eval is: {acc}")
    # logging.info(f"F1 for OSA eval is: {f1}")
    # logging.info(f"Recall for OSA eval is: {recall}")

    logging.info("Done")


if __name__ == "__main__":

    main()
