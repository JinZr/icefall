#!/usr/bin/env bash

python local/forming_alignment_manifest.py 

python local/compute_fbank_osa.py

python local/fix_sup_after_cut_into_windows.py \
    --input-cuts ./data/fbank/osa_cuts_all.jsonl.gz \
    --output-cuts ./data/fbank/osa_cuts_all_windows_fixed.jsonl.gz 

python local/remove_cuts_without_sup.py \
    --input-cuts ./data/fbank/osa_cuts_all_windows_fixed.jsonl.gz \
    --output-cuts ./data/fbank/osa_cuts_all_windows_fixed_filtered_empty_sup.jsonl.gz 

python local/convert_into_at_style_sup.py \
    --input-cuts ./data/fbank/osa_cuts_all_windows_fixed_filtered_empty_sup.jsonl.gz \
    --output-cuts ./data/fbank/osa_cuts_all_windows_fixed_filtered_empty_sup_at_style.jsonl.gz 