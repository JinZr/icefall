#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4,5,6,7 ./zipformer/train.py \
    --world-size 4 \
    --exp-dir ./zipformer/exp-small-100-epochs-w-recorder \
    --max-duration 1500 \
    --enable-musan 0 \
    --num-epochs 100 \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --use-recorder 1