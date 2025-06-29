#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python ./wavlm_ctc/finetune.py \
    --train-csv ./data/SHI_batch_1_seg/SHI_batch_1_seg.csv \
    --valid-csv ./data/SHI_batch_2_seg/SHI_batch_2_seg.csv \
    --model-name ./wavlm_ctc/wavlm-base/ \
    --batch-size 32 \
    --exp-dir ./wavlm_ctc/exp_base/ 