#!/usr/bin/env bash

root_path=/mnt/nfs_share/jinzr/dat/tongren-snoring-children/20240718/wav_edf/edf/

for spkr in $(ls $root_path); do
    edf_path="${root_path}/${spkr}/${spkr}.edf"
    wav_path="${root_path}/${spkr}/${spkr}.wav"

    echo "Converting ${spkr}.edf to $wav_path"

    # python3 -m biosppy.signals.tools -i $edf_path -o $wav_path
    python local/edf_to_wav.py --input-edf $edf_path --output-wav $wav_path

    # sox ${wav_path} -r 16000 "${root_path}/${spkr}/${spkr}_16k.wav"
done