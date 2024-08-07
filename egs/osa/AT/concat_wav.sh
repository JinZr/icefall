#!/usr/bin/env bash

root_path=/mnt/nfs_share/jinzr/dat/tongren-snoring-children/20240516/wav_edf/snoring/

# data_dir=${root_path}/A48700-1527-0222/
# python local/concat_wav.py \
#     --output-wav ${data_dir}/A48700-1527-0222.wav \
#     ${data_dir}/240222_1.wav \
#     ${data_dir}/240222_2.wav

# sox ${data_dir}/A48700-1527-0222.wav -r 16000 ${data_dir}/A48700-1527-0222_16k.wav

# data_dir=${root_path}/A48782-1522-0302/
# python local/concat_wav.py \
#     --output-wav ${data_dir}/A48782-1522-0302.wav \
#     ${data_dir}/240302_1_1522.wav \
#     ${data_dir}/240302_2_1522.wav

# sox ${data_dir}/A48782-1522-0302.wav -r 16000 ${data_dir}/A48782-1522-0302_16k.wav

# data_dir=${root_path}/A48809-1527-0306/
# python local/concat_wav.py \
#     --output-wav ${data_dir}/A48809-1527-0306.wav \
#     ${data_dir}/240306_1.wav \
#     ${data_dir}/240306_2.wav

# sox ${data_dir}/A48809-1527-0306.wav -r 16000 ${data_dir}/A48809-1527-0306_16k.wav

data_dir=${root_path}/A48837-1522-0309/
python local/concat_wav.py \
    --output-wav ${data_dir}/A48837-1522-0309.wav \
    ${data_dir}/240309_1_1522.wav \
    ${data_dir}/240309_2_1522.wav

sox ${data_dir}/A48837-1522-0309.wav -r 16000 ${data_dir}/A48837-1522-0309_16k.wav

data_dir=${root_path}/A48868-1527-0314/
python local/concat_wav.py \
    --output-wav ${data_dir}/A48868-1527-0314.wav \
    ${data_dir}/240314_1.wav \
    ${data_dir}/240314_2.wav

sox ${data_dir}/A48868-1527-0314.wav -r 16000 ${data_dir}/A48868-1527-0314_16k.wav

data_dir=${root_path}/A48869-1522-0314/
python local/concat_wav.py \
    --output-wav ${data_dir}/A48869-1522-0314.wav \
    ${data_dir}/240314_1_1522.wav \
    ${data_dir}/240314_2_1522.wav

sox ${data_dir}/A48869-1522-0314.wav -r 16000 ${data_dir}/A48869-1522-0314_16k.wav

data_dir=${root_path}/A49009-1526-0322/
python local/concat_wav.py \
    --output-wav ${data_dir}/A49009-1526-0322.wav \
    ${data_dir}/240322_2041.wav \
    ${data_dir}/240323_0327.wav

sox ${data_dir}/A49009-1526-0322.wav -r 16000 ${data_dir}/A49009-1526-0322_16k.wav

data_dir=${root_path}/A49010-1522-0322/
python local/concat_wav.py \
    --output-wav ${data_dir}/A49010-1522-0322.wav \
    ${data_dir}/240322_2237.wav \
    ${data_dir}/240323_0523.wav

sox ${data_dir}/A49010-1522-0322.wav -r 16000 ${data_dir}/A49010-1522-0322_16k.wav

data_dir=${root_path}/A49031-1527-0328/
python local/concat_wav.py \
    --output-wav ${data_dir}/A49031-1527-0328.wav \
    ${data_dir}/240328_1.wav \
    ${data_dir}/240328_2.wav

sox ${data_dir}/A49031-1527-0328.wav -r 16000 ${data_dir}/A49031-1527-0328_16k.wav

data_dir=${root_path}/A49031Z-1526-0326/
python local/concat_wav.py \
    --output-wav ${data_dir}/A49031Z-1526-0326.wav \
    ${data_dir}/240326_2113.wav \
    ${data_dir}/240327_0359.wav

sox ${data_dir}/A49031Z-1526-0326.wav -r 16000 ${data_dir}/A49031Z-1526-0326_16k.wav

data_dir=${root_path}/A49331-1522-0507/
python local/concat_wav.py \
    --output-wav ${data_dir}/A49331-1522-0507.wav \
    ${data_dir}/240507_2125.wav \
    ${data_dir}/240508_0411.wav

sox ${data_dir}/A49331-1522-0507.wav -r 16000 ${data_dir}/A49331-1522-0507_16k.wav

data_dir=${root_path}/A49358-1522-0509/
python local/concat_wav.py \
    --output-wav ${data_dir}/A49358-1522-0509.wav \
    ${data_dir}/240509_2146.wav \
    ${data_dir}/240510_0432.wav

sox ${data_dir}/A49358-1522-0509.wav -r 16000 ${data_dir}/A49358-1522-0509_16k.wav