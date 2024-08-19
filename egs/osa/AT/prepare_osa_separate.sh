#!/usr/bin/env bash

python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/A49009.csv \
    --output-dir ./data/manifests \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/snoring/A49009-1526-0322/A49009-1526-0322_16k.wav \
    --offset 1387

python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/A49010.csv \
    --output-dir ./data/manifests \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/snoring/A49010-1522-0322/A49010-1522-0322_16k.wav \
    --offset 5364

python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/A49265.csv \
    --output-dir ./data/manifests \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/snoring/A49265-1527-0426/A49265-1527-0426_16k.wav \
    --offset -141

python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/A49266.csv \
    --output-dir ./data/manifests \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/snoring/A49266-1522-0426/A49266-1522-0426_16k.wav \
    --offset 607

python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/A49268.csv \
    --output-dir ./data/manifests \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/snoring/A49268-1528-0426/A49268-1528-0426_16k.wav \
    --offset -175

python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/A49270.csv \
    --output-dir ./data/manifests \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/snoring/A49270-1526-0426/A49270-1526-0426_16k.wav \
    --offset 13

python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/A49313.csv \
    --output-dir ./data/manifests \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/snoring/A49313-1527-0506/A49313-1527-0506_16k.wav \
    --offset -808

python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/A49316.csv \
    --output-dir ./data/manifests \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/snoring/A49316-1526-0506/A49316-1526-0506_16k.wav \
    --offset 149

python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/A49331.csv \
    --output-dir ./data/manifests \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/snoring/A49331-1522-0507/A49331-1522-0507_16k.wav \
    --offset -942

python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/A49345.csv \
    --output-dir ./data/manifests \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/snoring/A49345-1527-0508/A49345-1527-0508_16k.wav \
    --offset 1900

python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/A49358.csv \
    --output-dir ./data/manifests \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/snoring/A49358-1522-0509/A49358-1522-0509_16k.wav \
    --offset 6874

python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/B47602.csv \
    --output-dir ./data/manifests \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/snoring/B47602-1526-0427/B47602-1526-0427_16k.wav \
    --offset 3668

lhotse combine \
    ./data/manifests/osa_recordings_A*.jsonl.gz \
    ./data/manifests/osa_recordings_B47602.jsonl.gz \
    ./data/manifests/osa_recordings_recorder.jsonl.gz

lhotse combine \
    ./data/manifests/osa_supervisions_A*.jsonl.gz \
    ./data/manifests/osa_supervisions_B47602.jsonl.gz \
    ./data/manifests/osa_supervisions_recorder.jsonl.gz

python local/compute_fbank_osa_recorder.py

python local/fix_sup_after_cut_into_windows.py \
    --input-cuts ./data/fbank/osa_cuts_recorder.jsonl.gz \
    --output-cuts ./data/fbank/osa_cuts_recorder_windows_fixed.jsonl.gz 

python local/remove_cuts_without_sup.py \
    --input-cuts ./data/fbank/osa_cuts_recorder_windows_fixed.jsonl.gz \
    --output-cuts ./data/fbank/osa_cuts_recorder_windows_fixed_filtered_empty_sup.jsonl.gz 

python local/convert_into_at_style_sup.py \
    --input-cuts ./data/fbank/osa_cuts_recorder_windows_fixed_filtered_empty_sup.jsonl.gz \
    --output-cuts ./data/fbank/osa_cuts_recorder_windows_fixed_filtered_empty_sup_at_style.jsonl.gz 

cat <(gunzip -c ./data/fbank/osa_cuts_recorder_windows_fixed_filtered_empty_sup_at_style.jsonl.gz ) | \
      shuf | gzip -c > data/fbank/osa_cuts_recorder_windows_fixed_filtered_empty_sup_at_style-shuf.jsonl.gz
    
lhotse subset --first 1513 \
    data/fbank/osa_cuts_recorder_windows_fixed_filtered_empty_sup_at_style-shuf.jsonl.gz \
    data/fbank/test_recorder.jsonl.gz

lhotse subset --last 34000 \
    data/fbank/osa_cuts_recorder_windows_fixed_filtered_empty_sup_at_style-shuf.jsonl.gz \
    data/fbank/train_recorder.jsonl.gz