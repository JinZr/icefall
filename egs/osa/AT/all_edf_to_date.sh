#!/usr/bin/env bash

root_path=/mnt/nfs_share/jinzr/dat/tongren-snoring-children/20240516/edf

for spkr in $(ls $root_path); do
    edf_path="${root_path}/${spkr}/${spkr}.edf"

    echo "Reading date from ${spkr}.edf"

    python local/edf_to_date.py --input-edf $edf_path 
done
