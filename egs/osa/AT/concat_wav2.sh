#!/usr/bin/env bash

root_path=/mnt/nfs_share/jinzr/dat/tongren-snoring-children/20240718/wav_edf/wav/

data_dir=${root_path}/A49698-1527-0614/
python local/concat_wav.py \
    --output-wav ${data_dir}/A49698-1527-0614.wav \
    ${data_dir}/240614_2119.wav \
    ${data_dir}/240615_0405.wav
sox ${data_dir}/A49698-1527-0614.wav -r 16000 ${data_dir}/A49698-1527-0614_16k.wav

data_dir=${root_path}/A49760-1522-0621/
python local/concat_wav.py \
    --output-wav ${data_dir}/A49760-1522-0621.wav \
    ${data_dir}/240621_2036.wav \
    ${data_dir}/240622_0321.wav
sox ${data_dir}/A49760-1522-0621.wav -r 16000 ${data_dir}/A49760-1522-0621_16k.wav

data_dir=${root_path}/A49762-1528-0621/
python local/concat_wav.py \
    --output-wav ${data_dir}/A49762-1528-0621.wav \
    ${data_dir}/240621_2115.wav \
    ${data_dir}/240622_0401.wav
sox ${data_dir}/A49762-1528-0621.wav -r 16000 ${data_dir}/A49762-1528-0621_16k.wav

data_dir=${root_path}/A49765-1527-0622/
python local/concat_wav.py \
    --output-wav ${data_dir}/A49765-1527-0622.wav \
    ${data_dir}/240622_2149.wav \
    ${data_dir}/240623_0435.wav
sox ${data_dir}/A49765-1527-0622.wav -r 16000 ${data_dir}/A49765-1527-0622_16k.wav

data_dir=${root_path}/A49794-1522-0625/
python local/concat_wav.py \
    --output-wav ${data_dir}/A49794-1522-0625.wav \
    ${data_dir}/240625_2057.wav \
    ${data_dir}/240626_0343.wav
sox ${data_dir}/A49794-1522-0625.wav -r 16000 ${data_dir}/A49794-1522-0625_16k.wav

data_dir=${root_path}/A49881-1527-0703/
python local/concat_wav.py \
    --output-wav ${data_dir}/A49881-1527-0703.wav \
    ${data_dir}/240703_2128.wav \
    ${data_dir}/240704_0414.wav
sox ${data_dir}/A49881-1527-0703.wav -r 16000 ${data_dir}/A49881-1527-0703_16k.wav

sid=A49892-1527-0704
data_dir=${root_path}/${sid}/
python local/concat_wav.py \
    --output-wav ${data_dir}/${sid}.wav \
    ${data_dir}/240704_2024.wav \
    ${data_dir}/240705_0312.wav
sox ${data_dir}/${sid}.wav -r 16000 ${data_dir}/${sid}_16k.wav

sid=A49899-1527-0705
data_dir=${root_path}/${sid}/
python local/concat_wav.py \
    --output-wav ${data_dir}/${sid}.wav \
    ${data_dir}/240705_2132.wav \
    ${data_dir}/240706_0418.wav
sox ${data_dir}/${sid}.wav -r 16000 ${data_dir}/${sid}_16k.wav

sid=A49977-1526-0713
data_dir=${root_path}/${sid}/
python local/concat_wav.py \
    --output-wav ${data_dir}/${sid}.wav \
    ${data_dir}/240713_2114.wav \
    ${data_dir}/240714_0400.wav
sox ${data_dir}/${sid}.wav -r 16000 ${data_dir}/${sid}_16k.wav

sid=A49980-1527-0713
data_dir=${root_path}/${sid}/
python local/concat_wav.py \
    --output-wav ${data_dir}/${sid}.wav \
    ${data_dir}/240713_2117.wav \
    ${data_dir}/240714_0403.wav
sox ${data_dir}/${sid}.wav -r 16000 ${data_dir}/${sid}_16k.wav

sid=B48380-1522-0617
data_dir=${root_path}/${sid}/
python local/concat_wav.py \
    --output-wav ${data_dir}/${sid}.wav \
    ${data_dir}/240614_2114.wav \
    ${data_dir}/240615_0359.wav
sox ${data_dir}/${sid}.wav -r 16000 ${data_dir}/${sid}_16k.wav