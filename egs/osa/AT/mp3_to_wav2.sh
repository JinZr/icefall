#!/usr/bin/env bash

root_path=/home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/wav/

data_dir=${root_path}/A49437-1522-0516/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/240516_2107.mp3 \
    --output-wav ${data_dir}/A49437-1522-0516.wav 
sox ${data_dir}/A49437-1522-0516.wav -r 16000 ${data_dir}/A49437-1522-0516_16k.wav

data_dir=${root_path}/A49451-1526-0517/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/240517_2059.mp3 \
    --output-wav ${data_dir}/A49451-1526-0517.wav 
sox ${data_dir}/A49451-1526-0517.wav -r 16000 ${data_dir}/A49451-1526-0517_16k.wav

data_dir=${root_path}/A49452-1522-0517/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/240517_2110.mp3 \
    --output-wav ${data_dir}/A49452-1522-0517.wav 
sox ${data_dir}/A49452-1522-0517.wav -r 16000 ${data_dir}/A49452-1522-0517_16k.wav

data_dir=${root_path}/A49494-1527-0522/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/240522_2048.mp3 \
    --output-wav ${data_dir}/A49494-1527-0522.wav 
sox ${data_dir}/A49494-1527-0522.wav -r 16000 ${data_dir}/A49494-1527-0522_16k.wav

data_dir=${root_path}/A49495-1526-0522/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/240522_2045.mp3 \
    --output-wav ${data_dir}/A49495-1526-0522.wav 
sox ${data_dir}/A49495-1526-0522.wav -r 16000 ${data_dir}/A49495-1526-0522_16k.wav

data_dir=${root_path}/A49511-1527-0524/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/240524_2030.mp3 \
    --output-wav ${data_dir}/A49511-1527-0524.wav 
sox ${data_dir}/A49511-1527-0524.wav -r 16000 ${data_dir}/A49511-1527-0524_16k.wav

data_dir=${root_path}/A49525-1527-0527/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/240527_2057.mp3 \
    --output-wav ${data_dir}/A49525-1527-0527.wav 
sox ${data_dir}/A49525-1527-0527.wav -r 16000 ${data_dir}/A49525-1527-0527_16k.wav

data_dir=${root_path}/A49589-1526-0601/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/240601_2211.mp3 \
    --output-wav ${data_dir}/A49589-1526-0601.wav 
sox ${data_dir}/A49589-1526-0601.wav -r 16000 ${data_dir}/A49589-1526-0601_16k.wav

data_dir=${root_path}/A49591-1528-0601/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/240601_2209.mp3 \
    --output-wav ${data_dir}/A49591-1528-0601.wav 
sox ${data_dir}/A49591-1528-0601.wav -r 16000 ${data_dir}/A49591-1528-0601_16k.wav

data_dir=${root_path}/A49604-1527-0603/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/240603_2010.mp3 \
    --output-wav ${data_dir}/A49604-1527-0603.wav 
sox ${data_dir}/A49604-1527-0603.wav -r 16000 ${data_dir}/A49604-1527-0603_16k.wav

data_dir=${root_path}/A49606-1526-0603/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/240603_2013.mp3 \
    --output-wav ${data_dir}/A49606-1526-0603.wav 
sox ${data_dir}/A49606-1526-0603.wav -r 16000 ${data_dir}/A49606-1526-0603_16k.wav

data_dir=${root_path}/A49621-1526-0604/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/240604_nnnn.mp3 \
    --output-wav ${data_dir}/A49621-1526-0604.wav 
sox ${data_dir}/A49621-1526-0604.wav -r 16000 ${data_dir}/A49621-1526-0604_16k.wav

data_dir=${root_path}/A49630-1527-0605/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/240605_2113.mp3 \
    --output-wav ${data_dir}/A49630-1527-0605.wav 
sox ${data_dir}/A49630-1527-0605.wav -r 16000 ${data_dir}/A49630-1527-0605_16k.wav

data_dir=${root_path}/A49671-1527-0611/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/240611_2011.mp3 \
    --output-wav ${data_dir}/A49671-1527-0611.wav 
sox ${data_dir}/A49671-1527-0611.wav -r 16000 ${data_dir}/A49671-1527-0611_16k.wav

sid=A49697-1526-0614
fname=240614_2120.mp3
data_dir=${root_path}/${sid}/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/${fname} \
    --output-wav ${data_dir}/${sid}.wav 
sox ${data_dir}/${sid}.wav -r 16000 ${data_dir}/${sid}_16k.wav

sid=A49759-1526-0621
fname=240621_2038.mp3
data_dir=${root_path}/${sid}/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/${fname} \
    --output-wav ${data_dir}/${sid}.wav 
sox ${data_dir}/${sid}.wav -r 16000 ${data_dir}/${sid}_16k.wav

sid=A49798-1526-0625
fname=240625_2059.mp3
data_dir=${root_path}/${sid}/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/${fname} \
    --output-wav ${data_dir}/${sid}.wav 
sox ${data_dir}/${sid}.wav -r 16000 ${data_dir}/${sid}_16k.wav

sid=A49806-1527-0626
fname=240626_2108.mp3
data_dir=${root_path}/${sid}/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/${fname} \
    --output-wav ${data_dir}/${sid}.wav 
sox ${data_dir}/${sid}.wav -r 16000 ${data_dir}/${sid}_16k.wav

sid=A49822-1526-0627
fname=240627_2247.mp3
data_dir=${root_path}/${sid}/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/${fname} \
    --output-wav ${data_dir}/${sid}.wav 
sox ${data_dir}/${sid}.wav -r 16000 ${data_dir}/${sid}_16k.wav

sid=A49835-1522-0628
fname=240628_2055.mp3
data_dir=${root_path}/${sid}/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/${fname} \
    --output-wav ${data_dir}/${sid}.wav 
sox ${data_dir}/${sid}.wav -r 16000 ${data_dir}/${sid}_16k.wav

sid=A49836-1527-0628
fname=240628_2059.mp3
data_dir=${root_path}/${sid}/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/${fname} \
    --output-wav ${data_dir}/${sid}.wav 
sox ${data_dir}/${sid}.wav -r 16000 ${data_dir}/${sid}_16k.wav

sid=A49841-1522-0629
fname=240629_2053.mp3
data_dir=${root_path}/${sid}/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/${fname} \
    --output-wav ${data_dir}/${sid}.wav 
sox ${data_dir}/${sid}.wav -r 16000 ${data_dir}/${sid}_16k.wav

sid=A49896-1526-0705
fname=240705_2129.mp3
data_dir=${root_path}/${sid}/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/${fname} \
    --output-wav ${data_dir}/${sid}.wav 
sox ${data_dir}/${sid}.wav -r 16000 ${data_dir}/${sid}_16k.wav

sid=A49948-1522-0710
fname=240710_2031.mp3
data_dir=${root_path}/${sid}/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/${fname} \
    --output-wav ${data_dir}/${sid}.wav 
sox ${data_dir}/${sid}.wav -r 16000 ${data_dir}/${sid}_16k.wav

sid=A49979-1522-0713
fname=240713_2115.mp3
data_dir=${root_path}/${sid}/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/${fname} \
    --output-wav ${data_dir}/${sid}.wav 
sox ${data_dir}/${sid}.wav -r 16000 ${data_dir}/${sid}_16k.wav

sid=B48091-1522-0702
fname=240702_1959.mp3
data_dir=${root_path}/${sid}/
python local/mp3_to_wav.py \
    --input-mp3 ${data_dir}/${fname} \
    --output-wav ${data_dir}/${sid}.wav 
sox ${data_dir}/${sid}.wav -r 16000 ${data_dir}/${sid}_16k.wav
