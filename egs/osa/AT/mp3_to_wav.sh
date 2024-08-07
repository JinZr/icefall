#!/usr/bin/env bash

root_path=/mnt/nfs_share/jinzr/dat/tongren-snoring-children/20240516/wav_edf/snoring/

data_dir=${root_path}/A49243-1523-0424/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/240424_1723_1117患者.mp3 \
    --output-wav ${data_dir}/A49243-1523-0424.wav 
sox ${data_dir}/A49243-1523-0424.wav -r 16000 ${data_dir}/A49243-1523-0424_16k.wav

data_dir=${root_path}/A49266-1522-0426/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/240426_2017.mp3 \
    --output-wav ${data_dir}/A49266-1522-0426.wav 
sox ${data_dir}/A49266-1522-0426.wav -r 16000 ${data_dir}/A49266-1522-0426_16k.wav

data_dir=${root_path}/A49270-1526-0426/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/240426_2056.mp3 \
    --output-wav ${data_dir}/A49270-1526-0426.wav 
sox ${data_dir}/A49270-1526-0426.wav -r 16000 ${data_dir}/A49270-1526-0426_16k.wav

data_dir=${root_path}/A49345-1527-0508/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/240508_2140.mp3 \
    --output-wav ${data_dir}/A49345-1527-0508.wav 
sox ${data_dir}/A49345-1527-0508.wav -r 16000 ${data_dir}/A49345-1527-0508_16k.wav

data_dir=${root_path}/A49244-1522-0424/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/240424_1723（1112）.mp3 \
    --output-wav ${data_dir}/A49244-1522-0424.wav 
sox ${data_dir}/A49244-1522-0424.wav -r 16000 ${data_dir}/A49244-1522-0424_16k.wav

data_dir=${root_path}/A49268-1528-0426/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/240426_2049.mp3 \
    --output-wav ${data_dir}/A49268-1528-0426.wav 
sox ${data_dir}/A49268-1528-0426.wav -r 16000 ${data_dir}/A49268-1528-0426_16k.wav

data_dir=${root_path}/A49313-1527-0506/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/240506_2026.mp3 \
    --output-wav ${data_dir}/A49313-1527-0506.wav 
sox ${data_dir}/A49313-1527-0506.wav -r 16000 ${data_dir}/A49313-1527-0506_16k.wav

data_dir=${root_path}/B47602-1526-0427/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/240427_2150.mp3 \
    --output-wav ${data_dir}/B47602-1526-0427.wav 
sox ${data_dir}/B47602-1526-0427.wav -r 16000 ${data_dir}/B47602-1526-0427_16k.wav

data_dir=${root_path}/A49265-1527-0426/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/240426_1956.mp3 \
    --output-wav ${data_dir}/A49265-1527-0426.wav 
sox ${data_dir}/A49265-1527-0426.wav -r 16000 ${data_dir}/A49265-1527-0426_16k.wav

data_dir=${root_path}/A49269-1523-0426/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/240426_2009_1229患者.mp3 \
    --output-wav ${data_dir}/A49269-1523-0426.wav 
sox ${data_dir}/A49269-1523-0426.wav -r 16000 ${data_dir}/A49269-1523-0426_16k.wav

data_dir=${root_path}/A49316-1526-0506/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/240506_2024.mp3 \
    --output-wav ${data_dir}/A49316-1526-0506.wav 
sox ${data_dir}/A49316-1526-0506.wav -r 16000 ${data_dir}/A49316-1526-0506_16k.wav

