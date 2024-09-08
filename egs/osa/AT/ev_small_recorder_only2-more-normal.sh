#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4 ./zipformer/evaluate.py \
    --exp-dir ./zipformer/exp-small-100-epochs-recorder-only-w-batch2-more-normal \
    --max-duration 1500 \
    --epoch 100 \
    --avg 40 \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --use-recorder 1 \
    --use-attached 0 \
    --threshold 0.6

for threshold in 0.65 0.7 0.75 0.8 0.85 0.9 0.95; do
    CUDA_VISIBLE_DEVICES=4 ./zipformer/evaluate.py \
        --exp-dir ./zipformer/exp-small-100-epochs-recorder-only-w-batch2-more-normal \
        --max-duration 1500 \
        --epoch 100 \
        --avg 40 \
        --num-encoder-layers 2,2,2,2,2,2 \
        --feedforward-dim 512,768,768,768,768,768 \
        --encoder-dim 192,256,256,256,256,256 \
        --encoder-unmasked-dim 192,192,192,192,192,192 \
        --use-recorder 1 \
        --use-attached 0 \
        --threshold $threshold
done

for threshold in 0.5 0.55; do
    CUDA_VISIBLE_DEVICES=4 ./zipformer/evaluate.py \
        --exp-dir ./zipformer/exp-small-100-epochs-recorder-only-w-batch2-more-normal \
        --max-duration 1500 \
        --epoch 100 \
        --avg 40 \
        --num-encoder-layers 2,2,2,2,2,2 \
        --feedforward-dim 512,768,768,768,768,768 \
        --encoder-dim 192,256,256,256,256,256 \
        --encoder-unmasked-dim 192,192,192,192,192,192 \
        --use-recorder 1 \
        --use-attached 0 \
        --threshold $threshold
done