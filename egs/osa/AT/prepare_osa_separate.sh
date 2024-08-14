#!/usr/bin/env bash

python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/A49009.csv \
    --output-dir ./data/manifests \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/snoring/A49009-1526-0322/A49009-1526-0322_16k.wav \
    --offset 1380

python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/A49009.csv \
    --output-dir ./data/manifests \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/snoring/A49009-1526-0322/A49009-1526-0322_16k.wav \
    --offset 1380