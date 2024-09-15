#!/usr/bin/env bash

for epoch in 70 80 90 100 110 120 130 140 150; do
	for avg in 10 20 30 40 50 60 70 80 90; do
        CUDA_VISIBLE_DEVICES=0 ./zipformer/evaluate.py \
            --exp-dir ./zipformer/exp-small-100-epochs-recorder-only-w-batch2-5s \
            --max-duration 1500 \
            --manifest-dir ./data/fbank_5s/ \
            --epoch $epoch \
            --avg $avg \
            --num-encoder-layers 2,2,2,2,2,2 \
            --feedforward-dim 512,768,768,768,768,768 \
            --encoder-dim 192,256,256,256,256,256 \
            --encoder-unmasked-dim 192,192,192,192,192,192 \
            --use-recorder 1 \
            --threshold 0.6 \
            --use-attached 0
    done
done

# epoch-120-avg-90 best
