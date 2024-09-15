#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4,5,6,7 ./zipformer/train.py \
    --world-size 4 \
    --exp-dir ./zipformer/exp-small-100-epochs-recorder-only-w-batch2-5s \
    --max-duration 1000 \
    --enable-musan 0 \
    --manifest-dir ./data/fbank_5s/ \
    --num-epochs 150 \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --use-recorder 1 \
    --use-attached 0 \
    --base-lr 0.03

exit 0
CUDA_VISIBLE_DEVICES=4,5,6,7 ./zipformer/train.py \
    --world-size 4 \
    --exp-dir ./zipformer/exp-small-100-epochs-recorder-only-w-batch2-base-lr-0.02 \
    --max-duration 1500 \
    --enable-musan 0 \
    --num-epochs 100 \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --use-recorder 1 \
    --use-attached 0 \
    --master-port 9527 \
    --base-lr 0.02
