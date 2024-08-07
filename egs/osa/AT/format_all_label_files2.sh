#!/bin/bash

python local/format_supervision_txt.py \
    --input-txt /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/edf/A48700/A48700_PSG4_ScoredEvents_Export.txt \
    --output-csv ./format_csv/A48700.csv

python local/format_supervision_txt.py \
    --input-txt /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/edf/A48782/A48782_PSG4_ScoredEvents_Export.txt \
    --output-csv ./format_csv/A48782.csv
    
python local/format_supervision_txt.py \
    --input-txt /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/edf/A48809/A48809_PSG4_ScoredEvents_Export.txt \
    --output-csv ./format_csv/A48809.csv

python local/format_supervision_txt.py \
    --input-txt /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/edf/A48837/A48837_PSG4_ScoredEvents_Export.txt \
    --output-csv ./format_csv/A48837.csv

python local/format_supervision_txt.py \
    --input-txt /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/edf/A49010/A49010_PSG4_ScoredEvents_Export.txt \
    --output-csv ./format_csv/A49010.csv

python local/format_supervision_txt.py \
    --input-txt /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/edf/A49031/49031_PSG4_ScoredEvents_Export.txt \
    --output-csv ./format_csv/A49031.csv

python local/format_supervision_csv.py \
    --input-csv /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/edf/A49009/A49009EVENT.csv \
    --output-csv ./format_csv/A49009.csv