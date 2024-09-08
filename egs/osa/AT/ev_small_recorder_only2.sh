#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4 ./zipformer/evaluate.py \
    --exp-dir ./zipformer/exp-small-100-epochs-recorder-only-w-batch2 \
    --max-duration 1500 \
    --epoch 100 \
    --avg 50 \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --use-recorder 1 \
    --use-attached 0