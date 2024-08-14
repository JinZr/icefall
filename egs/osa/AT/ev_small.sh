#!/usr/bin/env bash

./zipformer/evaluate.py \
    --exp-dir ./zipformer/exp-small-100-epochs \
    --max-duration 1500 \
    --epoch 100 \
    --avg 50 \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 