#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python ./w2v_ctc/finetune.py \
    --train-csv ./data/SHI_batch_1_seg/SHI_batch_1_seg.csv \
    --valid-csv ./data/SHI_batch_2_seg/SHI_batch_2_seg.csv \
    --model-name ./w2v_ctc/wav2vec2-large/ \
    --batch-size 16 \
    --exp-dir ./w2v_ctc/exp_large/ 