#!/bin/bash

python local/format_supervision_txt_w_offset.py \
    --input-txt /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/edf/A49437/A49437.txt \
    --speaker-id A49437 \
    --offset 5175 \
    --output-csv ./format_csv_with_offset/A49437.csv

python local/format_supervision_csv_w_offset.py \
    --input-csv /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/edf/A49589/A49589.csv \
    --speaker-id A49589 \
    --offset 4395 \
    --output-csv ./format_csv_with_offset/A49589.csv

python local/format_supervision_csv_w_offset.py \
    --input-csv /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/edf/A49591/A49591.csv \
    --speaker-id A49591 \
    --offset 3563 \
    --output-csv ./format_csv_with_offset/A49591.csv

python local/format_supervision_txt_w_offset.py \
    --input-txt /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/edf/A49630/A49630.txt \
    --speaker-id A49630 \
    --offset 3799 \
    --output-csv ./format_csv_with_offset/A49630.csv

python local/format_supervision_csv_w_offset.py \
    --input-csv /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/edf/A49697/A49697.csv \
    --speaker-id A49697 \
    --offset 3795 \
    --output-csv ./format_csv_with_offset/A49697.csv

python local/format_supervision_txt_w_offset.py \
    --input-txt /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/edf/A49765/A49765.txt \
    --speaker-id A49765 \
    --offset 5202 \
    --output-csv ./format_csv_with_offset/A49765.csv

python local/format_supervision_txt_w_offset.py \
    --input-txt /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/edf/A49806/A49806.txt \
    --speaker-id A49806 \
    --offset 3045 \
    --output-csv ./format_csv_with_offset/A49806.csv

python local/format_supervision_csv_w_offset.py \
    --input-csv /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/edf/A49822/A49822.csv \
    --speaker-id A49822 \
    --offset 7851 \
    --output-csv ./format_csv_with_offset/A49822.csv

python local/format_supervision_csv_w_offset.py \
    --input-csv /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/edf/A49896/A49896.csv \
    --speaker-id A49896 \
    --offset 6518 \
    --output-csv ./format_csv_with_offset/A49896.csv

python local/format_supervision_csv_w_offset.py \
    --input-csv /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/edf/A49977/A49977.csv \
    --speaker-id A49977 \
    --offset 590 \
    --output-csv ./format_csv_with_offset/A49977.csv

python local/format_supervision_txt_w_offset.py \
    --input-txt /home/jinzr/nfs/dat/tongren-snoring-children/20240718/wav_edf/edf/B48380/B48380.txt \
    --speaker-id B48380 \
    --offset 3387 \
    --output-csv ./format_csv_with_offset/B48380.csv
