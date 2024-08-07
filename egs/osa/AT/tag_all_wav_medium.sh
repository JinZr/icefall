#!/usr/bin/env bash

root_path=/mnt/nfs_share/jinzr/dat/tongren-snoring-children/20240516/edf/

for spkr in $(ls ${root_path}); do
    python ./zipformer/pretrained_long_audio.py \
        --checkpoint ./zipformer/exp_at_as_full/pretrained.pt \
        --label-dict downloads/audioset/class_labels_indices.csv \
	--audio-chunk-size 6 \
        /mnt/nfs_share/jinzr/dat/tongren-snoring-children/20240516/edf/${spkr}/${spkr}_16k.wav
done
