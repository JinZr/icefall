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

python local/format_supervision_txt.py \
    --input-txt /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/edf/A48868/A48868_PSG4_StudyLog_Export.txt \
    --output-csv ./format_csv/A48868.csv

python local/format_supervision_txt.py \
    --input-txt /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/edf/A48869/48869_PSG4_ScoredEvents_Export.txt \
    --output-csv ./format_csv/A48869.csv

python local/format_supervision_xls.py \
    --input-xls /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/edf/A49243/A49243.xls \
    --output-csv ./format_csv/A49243.csv

python local/format_supervision_xls.py \
    --input-xls /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/edf/A49244/A49244.xls \
    --output-csv ./format_csv/A49244.csv

python local/format_supervision_txt.py \
    --input-txt /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/edf/A49265/49265_PSG4_ScoredEvents_Export.txt \
    --output-csv ./format_csv/A49265.csv

python local/format_supervision_txt.py \
    --input-txt /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/edf/A49266/A49266_PSG4_ScoredEvents_Export.txt \
    --output-csv ./format_csv/A49266.csv

python local/format_supervision_csv.py \
    --input-csv /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/edf/A49268/A49268.csv \
    --output-csv ./format_csv/A49268.csv

python local/format_supervision_csv.py \
    --input-csv /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/edf/A49270/A490270EVENT.csv \
    --output-csv ./format_csv/A49270.csv

python local/format_supervision_txt.py \
    --input-txt /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/edf/A49313/A49313_PSG4_ScoredEvents_Export.txt \
    --output-csv ./format_csv/A49313.csv

python local/format_supervision_csv.py \
    --input-csv /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/edf/A49316/A49316EVENT.csv \
    --output-csv ./format_csv/A49316.csv

python local/format_supervision_txt.py \
    --input-txt /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/edf/A49331/A49331-PSG4_ScoredEvents_Export.txt \
    --output-csv ./format_csv/A49331.csv

python local/format_supervision_txt.py \
    --input-txt /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/edf/A49345/A49345_PSG4_ScoredEvents_Export.txt \
    --output-csv ./format_csv/A49345.csv

python local/format_supervision_txt.py \
    --input-txt /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/edf/A49358/A49358_PSG4_ScoredEvents_Export.txt \
    --output-csv ./format_csv/A49358.csv

python local/format_supervision_csv.py \
    --input-csv /home/jinzr/nfs/dat/tongren-snoring-children/20240516/wav_edf/edf/B47602/B47602EVENT.csv \
    --output-csv ./format_csv/B47602.csv