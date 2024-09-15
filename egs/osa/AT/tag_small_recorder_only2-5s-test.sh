#!/usr/bin/env bash

threshold=0.575

echo '======== Threshold: ========'
echo ${threshold}
echo '======== Severe Group (2) ========'

python ./zipformer/pretrained_long_audio_ft.py \
    --checkpoint ./zipformer/exp-small-100-epochs-recorder-only-w-batch2-5s/pretrained.pt \
    --label-dict downloads/osa/class_labels_indices.csv \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --audio-chunk-size 5 \
    --num-events 2 \
    --duration 14.5 \
    --threshold ${threshold} \
    /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/A49806-1527-0626/A49806-1527-0626_16k.mp3

python ./zipformer/pretrained_long_audio_ft.py \
    --checkpoint ./zipformer/exp-small-100-epochs-recorder-only-w-batch2-5s/pretrained.pt \
    --label-dict downloads/osa/class_labels_indices.csv \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --audio-chunk-size 5 \
    --num-events 2 \
    --duration 9.5 \
    --threshold ${threshold} \
    /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/A49977-1526-0713/A49977-1526-0713_16k.wav


echo '======== Mild Group (6) ========'

python ./zipformer/pretrained_long_audio_ft.py \
    --checkpoint ./zipformer/exp-small-100-epochs-recorder-only-w-batch2-5s/pretrained.pt \
    --label-dict downloads/osa/class_labels_indices.csv \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --audio-chunk-size 5 \
    --num-events 2 \
    --duration 10.75 \
    --threshold ${threshold} \
    /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/A49437-1522-0516/A49437-1522-0516_16k.wav

python ./zipformer/pretrained_long_audio_ft.py \
    --checkpoint ./zipformer/exp-small-100-epochs-recorder-only-w-batch2-5s/pretrained.pt \
    --label-dict downloads/osa/class_labels_indices.csv \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --audio-chunk-size 5 \
    --num-events 2 \
    --duration 9.63 \
    --threshold ${threshold} \
    /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/A49589-1526-0601/A49589-1526-0601_16k.wav

python ./zipformer/pretrained_long_audio_ft.py \
    --checkpoint ./zipformer/exp-small-100-epochs-recorder-only-w-batch2-5s/pretrained.pt \
    --label-dict downloads/osa/class_labels_indices.csv \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --audio-chunk-size 5 \
    --num-events 2 \
    --duration 8.65 \
    --threshold ${threshold} \
    /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/A49591-1528-0601/A49591-1528-0601_16k.wav

python ./zipformer/pretrained_long_audio_ft.py \
    --checkpoint ./zipformer/exp-small-100-epochs-recorder-only-w-batch2-5s/pretrained.pt \
    --label-dict downloads/osa/class_labels_indices.csv \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --audio-chunk-size 5 \
    --num-events 2 \
    --duration 9.15 \
    --threshold ${threshold} \
    /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/A49759-1526-0621/A49759-1526-0621_16k.wav

python ./zipformer/pretrained_long_audio_ft.py \
    --checkpoint ./zipformer/exp-small-100-epochs-recorder-only-w-batch2-5s/pretrained.pt \
    --label-dict downloads/osa/class_labels_indices.csv \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --audio-chunk-size 5 \
    --num-events 2 \
    --duration 9.5 \
    --threshold ${threshold} \
    /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/A49822-1526-0627/A49822-1526-0627_16k.wav

python ./zipformer/pretrained_long_audio_ft.py \
    --checkpoint ./zipformer/exp-small-100-epochs-recorder-only-w-batch2-5s/pretrained.pt \
    --label-dict downloads/osa/class_labels_indices.csv \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --audio-chunk-size 5 \
    --num-events 2 \
    --duration 9.5 \
    --threshold ${threshold} \
    /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/A49980-1527-0713/A49980-1527-0713_16k.wav

echo '======== Healthy Group (6) ========'

python ./zipformer/pretrained_long_audio_ft.py \
    --checkpoint ./zipformer/exp-small-100-epochs-recorder-only-w-batch2-5s/pretrained.pt \
    --label-dict downloads/osa/class_labels_indices.csv \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --audio-chunk-size 5 \
    --num-events 2 \
    --duration 10 \
    --threshold ${threshold} \
    /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/A49621-1526-0604/A49621-1526-0604_16k.wav

python ./zipformer/pretrained_long_audio_ft.py \
    --checkpoint ./zipformer/exp-small-100-epochs-recorder-only-w-batch2-5s/pretrained.pt \
    --label-dict downloads/osa/class_labels_indices.csv \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --audio-chunk-size 5 \
    --num-events 2 \
    --duration 10.50 \
    --threshold ${threshold} \
    /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/A49630-1527-0605/A49630-1527-0605_16k.wav

python ./zipformer/pretrained_long_audio_ft.py \
    --checkpoint ./zipformer/exp-small-100-epochs-recorder-only-w-batch2-5s/pretrained.pt \
    --label-dict downloads/osa/class_labels_indices.csv \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --audio-chunk-size 5 \
    --num-events 2 \
    --duration 10.35 \
    --threshold ${threshold} \
    /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/A49697-1526-0614/A49697-1526-0614_16k.wav

python ./zipformer/pretrained_long_audio_ft.py \
    --checkpoint ./zipformer/exp-small-100-epochs-recorder-only-w-batch2-5s/pretrained.pt \
    --label-dict downloads/osa/class_labels_indices.csv \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --audio-chunk-size 5 \
    --num-events 2 \
    --duration 10.27 \
    --threshold ${threshold} \
    /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/A49765-1527-0622/A49765-1527-0622_16k.wav

python ./zipformer/pretrained_long_audio_ft.py \
    --checkpoint ./zipformer/exp-small-100-epochs-recorder-only-w-batch2-5s/pretrained.pt \
    --label-dict downloads/osa/class_labels_indices.csv \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --audio-chunk-size 5 \
    --num-events 2 \
    --duration 9.5 \
    --threshold ${threshold} \
    /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/A49896-1526-0705/A49896-1526-0705_16k.wav

python ./zipformer/pretrained_long_audio_ft.py \
    --checkpoint ./zipformer/exp-small-100-epochs-recorder-only-w-batch2-5s/pretrained.pt \
    --label-dict downloads/osa/class_labels_indices.csv \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --audio-chunk-size 5 \
    --num-events 2 \
    --duration 9.5 \
    --threshold ${threshold} \
    /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/B48380-1522-0617/B48380-1522-0617_16k.wav
