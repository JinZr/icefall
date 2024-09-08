#!/usr/bin/env bash

./zipformer/export.py \
    --exp-dir ./zipformer/exp-small-100-epochs-recorder-only-w-batch2-more-normal \
    --epoch 100 \
    --avg 40 \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 