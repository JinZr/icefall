#!/usr/bin/env bash

./zipformer/train.py \
    --world-size 4 \
    --exp-dir ./zipformer/exp-small \
    --max-duration 1500 \
    --enable-musan 0 \
    --num-epochs 60 \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 

./zipformer/train.py \
    --world-size 4 \
    --exp-dir ./zipformer/exp-small-100-epochs \
    --max-duration 1500 \
    --enable-musan 0 \
    --num-epochs 100 \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 