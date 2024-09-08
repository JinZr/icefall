#!/usr/bin/env bash

python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/B48091.csv \
    --output-dir ./data/manifests_batch2 \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/B48091-1522-0702/B48091-1522-0702_16k.wav \
    --offset -277


python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/A49451.csv \
    --output-dir ./data/manifests_batch2 \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/A49451-1526-0517/A49451-1526-0517_16k.wav \
    --offset 2261

python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/A49452.csv \
    --output-dir ./data/manifests_batch2 \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/A49452-1522-0517/A49452-1522-0517_16k.wav \
    --offset 219

python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/A49495.csv \
    --output-dir ./data/manifests_batch2 \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/A49495-1526-0522/A49495-1526-0522_16k.wav \
    --offset 1004

python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/A49511.csv \
    --output-dir ./data/manifests_batch2 \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/A49511-1527-0524/A49511-1527-0524_16k.wav \
    --offset -1491

python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/A49525.csv \
    --output-dir ./data/manifests_batch2 \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/A49525-1527-0527/A49525-1527-0527_16k.wav \
    --offset -270

python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/A49604.csv \
    --output-dir ./data/manifests_batch2 \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/A49604-1527-0603/A49604-1527-0603_16k.wav \
    --offset 6

sid=A49606-1526-0603
python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/A49606.csv \
    --output-dir ./data/manifests_batch2 \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/${sid}/${sid}_16k.wav \
    --offset 132

sid=A49671-1527-0611
python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/A49671.csv \
    --output-dir ./data/manifests_batch2 \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/${sid}/${sid}_16k.wav \
    --offset -16


sid=A49698-1527-0614
python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/A49698.csv \
    --output-dir ./data/manifests_batch2 \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/${sid}/${sid}_16k.wav \
    --offset 1704

sid=A49760-1522-0621
fname=A49760
offset=1373
python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/${fname}.csv \
    --output-dir ./data/manifests_batch2 \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/${sid}/${sid}_16k.wav \
    --offset ${offset}

sid=A49762-1528-0621
fname=A49762
offset=297
python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/${fname}.csv \
    --output-dir ./data/manifests_batch2 \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/${sid}/${sid}_16k.wav \
    --offset ${offset}


sid=A49794-1522-0625
fname=A49794
offset=-35
python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/${fname}.csv \
    --output-dir ./data/manifests_batch2 \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/${sid}/${sid}_16k.wav \
    --offset ${offset}

sid=A49798-1526-0625
fname=A49798
offset=452
python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/${fname}.csv \
    --output-dir ./data/manifests_batch2 \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/${sid}/${sid}_16k.wav \
    --offset ${offset}


sid=A49835-1522-0628
fname=A49835
offset=-133
python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/${fname}.csv \
    --output-dir ./data/manifests_batch2 \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/${sid}/${sid}_16k.wav \
    --offset ${offset}

sid=A49836-1527-0628
fname=A49836
offset=1024
python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/${fname}.csv \
    --output-dir ./data/manifests_batch2 \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/${sid}/${sid}_16k.wav \
    --offset ${offset}

sid=A49881-1527-0703
fname=A49881
offset=-122
python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/${fname}.csv \
    --output-dir ./data/manifests_batch2 \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/${sid}/${sid}_16k.wav \
    --offset ${offset}

sid=A49892-1527-0704
fname=A49892
offset=-1545
python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/${fname}.csv \
    --output-dir ./data/manifests_batch2 \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/${sid}/${sid}_16k.wav \
    --offset ${offset}


sid=A49899-1527-0705
fname=A49899
offset=-118
python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/${fname}.csv \
    --output-dir ./data/manifests_batch2 \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/${sid}/${sid}_16k.wav \
    --offset ${offset}

sid=A49948-1522-0710
fname=A49948
offset=10
python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/${fname}.csv \
    --output-dir ./data/manifests_batch2 \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/${sid}/${sid}_16k.wav \
    --offset ${offset}

sid=A49979-1522-0713
fname=A49979
offset=1038
python local/forming_alignment_manifest_single.py \
    --csv-path ./format_csv/${fname}.csv \
    --output-dir ./data/manifests_batch2 \
    --audio-path /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/${sid}/${sid}_16k.wav \
    --offset ${offset}


lhotse combine \
    ./data/manifests_batch2/osa_recordings_A*.jsonl.gz \
    ./data/manifests_batch2/osa_recordings_B*.jsonl.gz \
    ./data/manifests_batch2/osa_recordings_recorder_batch2.jsonl.gz

lhotse combine \
    ./data/manifests_batch2/osa_supervisions_A*.jsonl.gz \
    ./data/manifests_batch2/osa_supervisions_B*.jsonl.gz \
    ./data/manifests_batch2/osa_supervisions_recorder_batch2.jsonl.gz

python local/compute_fbank_osa_recorder2.py

python local/fix_sup_after_cut_into_windows.py \
    --input-cuts ./data/fbank_batch2/osa_cuts_recorder_batch2.jsonl.gz \
    --output-cuts ./data/fbank_batch2/osa_cuts_recorder_batch2_windows_fixed.jsonl.gz 

python local/remove_cuts_without_sup_for_recorder2.py \
    --input-cuts ./data/fbank_batch2/osa_cuts_recorder_batch2_windows_fixed.jsonl.gz  \
    --output-dir ./data/fbank_batch2/

python local/convert_into_at_style_sup.py \
    --input-cuts ./data/fbank_batch2/osa_cuts_recorder_batch2_windows_fixed_filtered_empty_sup.jsonl.gz \
    --output-cuts ./data/fbank_batch2/osa_cuts_recorder_batch2_windows_fixed_filtered_empty_sup_at_style.jsonl.gz 

python local/convert_into_at_style_sup_for_recorder.py \
    --input-cuts ./data/fbank_batch2/osa_cuts_recorder_batch2_windows_fixed_should_be_normal.jsonl.gz \
    --output-cuts ./data/fbank_batch2/osa_cuts_recorder_batch2_windows_fixed_should_be_normal_at_style.jsonl.gz 

cat <(gunzip -c ./data/fbank_batch2/osa_cuts_recorder_batch2_windows_fixed_filtered_empty_sup_at_style.jsonl.gz ) | \
      shuf | gzip -c > data/fbank_batch2/osa_cuts_recorder_batch2_windows_fixed_filtered_empty_sup_at_style-shuf.jsonl.gz
    
cat <(gunzip -c ./data/fbank_batch2/osa_cuts_recorder_batch2_windows_fixed_should_be_normal_at_style.jsonl.gz ) | \
      shuf | gzip -c > data/fbank_batch2/osa_cuts_recorder_batch2_windows_fixed_should_be_normal_at_style-shuf.jsonl.gz
    

lhotse subset --first 14000 \
    data/fbank_batch2/osa_cuts_recorder_batch2_windows_fixed_should_be_normal_at_style-shuf.jsonl.gz \
    data/fbank_batch2/osa_cuts_recorder_batch2_windows_fixed_should_be_normal_at_style-shuf_first_14000.jsonl.gz

# lhotse subset --first 1513 \
#     data/fbank/osa_cuts_recorder_windows_fixed_filtered_empty_sup_at_style-shuf.jsonl.gz \
#     data/fbank/test_recorder.jsonl.gz

# lhotse subset --last 34000 \
#     data/fbank/osa_cuts_recorder_windows_fixed_filtered_empty_sup_at_style-shuf.jsonl.gz \
#     data/fbank/train_recorder.jsonl.gz

cat <(gunzip -c ./data/fbank/train_recorder-no-shuf.jsonl.gz ) | \
      shuf | gzip -c > data/fbank/train_recorder.jsonl.gz