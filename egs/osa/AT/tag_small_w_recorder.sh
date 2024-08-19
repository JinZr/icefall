#!/usr/bin/env bash

# root_path=/mnt/nfs_share/jinzr/dat/tongren-snoring-children/20240516/edf/

# for spkr in $(ls ${root_path}); do
#     python ./zipformer/pretrained_long_audio_ft.py \
#         --checkpoint ./zipformer/exp-small-100-epochs-w-recorder/pretrained.pt \
#         --label-dict downloads/osa/class_labels_indices.csv \
#         --num-encoder-layers 2,2,2,2,2,2 \
#         --feedforward-dim 512,768,768,768,768,768 \
#         --encoder-dim 192,256,256,256,256,256 \
#         --encoder-unmasked-dim 192,192,192,192,192,192 \
#         --audio-chunk-size 1 \
#         --num-events 2 \
#         /mnt/nfs_share/jinzr/dat/tongren-snoring-children/20240516/edf/${spkr}/${spkr}_16k.wav
# done

python ./zipformer/pretrained_long_audio_ft.py \
    --checkpoint ./zipformer/exp-small-100-epochs-w-recorder/pretrained.pt \
    --label-dict downloads/osa/class_labels_indices.csv \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --audio-chunk-size 1 \
    --num-events 2 \
    /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/snoring/A48700-1527-0222/A48700-1527-0222_16k.wav 

python ./zipformer/pretrained_long_audio_ft.py \
    --checkpoint ./zipformer/exp-small-100-epochs-w-recorder/pretrained.pt \
    --label-dict downloads/osa/class_labels_indices.csv \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --audio-chunk-size 1 \
    --num-events 2 \
    /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/snoring/A48782-1522-0302/A48782-1522-0302_16k.wav 

python ./zipformer/pretrained_long_audio_ft.py \
    --checkpoint ./zipformer/exp-small-100-epochs-w-recorder/pretrained.pt \
    --label-dict downloads/osa/class_labels_indices.csv \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --audio-chunk-size 1 \
    --num-events 2 \
    /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/snoring/A48809-1527-0306/A48809-1527-0306_16k.wav


python ./zipformer/pretrained_long_audio_ft.py \
    --checkpoint ./zipformer/exp-small-100-epochs-w-recorder/pretrained.pt \
    --label-dict downloads/osa/class_labels_indices.csv \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --audio-chunk-size 1 \
    --num-events 2 \
    /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/snoring/A48837-1522-0309/A48837-1522-0309_16k.wav

python ./zipformer/pretrained_long_audio_ft.py \
    --checkpoint ./zipformer/exp-small-100-epochs-w-recorder/pretrained.pt \
    --label-dict downloads/osa/class_labels_indices.csv \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --audio-chunk-size 1 \
    --num-events 2 \
    /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/snoring/A48868-1527-0314/A48868-1527-0314_16k.wav

python ./zipformer/pretrained_long_audio_ft.py \
    --checkpoint ./zipformer/exp-small-100-epochs-w-recorder/pretrained.pt \
    --label-dict downloads/osa/class_labels_indices.csv \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --audio-chunk-size 1 \
    --num-events 2 \
    /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/snoring/A48869-1522-0314/A48869-1522-0314_16k.wav

python ./zipformer/pretrained_long_audio_ft.py \
    --checkpoint ./zipformer/exp-small-100-epochs-w-recorder/pretrained.pt \
    --label-dict downloads/osa/class_labels_indices.csv \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --audio-chunk-size 1 \
    --num-events 2 \
    /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/snoring/A49031-1527-0328/A49031-1527-0328_16k.wav


