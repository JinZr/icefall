#!/usr/bin/env bash

root_path=/mnt/nfs_share/jinzr/dat/tongren-snoring-children/20240516/edf/

for spkr in $(ls ${root_path}); do
    python ./zipformer/pretrained_long_audio_ft.py \
        --checkpoint ./zipformer/exp-small/epoch-30.pt \
        --label-dict downloads/osa/class_labels_indices.csv \
        --num-encoder-layers 2,2,2,2,2,2 \
        --feedforward-dim 512,768,768,768,768,768 \
        --encoder-dim 192,256,256,256,256,256 \
        --encoder-unmasked-dim 192,192,192,192,192,192 \
        --audio-chunk-size 20 \
        --num-events 6 \
        /mnt/nfs_share/jinzr/dat/tongren-snoring-children/20240516/edf/${spkr}/${spkr}_16k.wav
done