for spkr in $(ls /home/jinzr/nfs/dat/tongren-snoring-children/20240516/edf/); do
    python local/format_supervision_xls.py \
        --input-xls /home/jinzr/nfs/dat/tongren-snoring-children/20240516/edf/"${spkr}"/"${spkr}".xls \
        --output-csv ./format_csv/"${spkr}".csv
done