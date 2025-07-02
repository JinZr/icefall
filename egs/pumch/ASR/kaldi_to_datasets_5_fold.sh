#!/usr/bin/env bash

for i in $(seq 1 5); do
    python local/kaldi_to_datasets.py ./data/5-folds/fold${i}/train/ --out ./data/5-folds/fold${i}/train.csv

    python local/kaldi_to_datasets.py ./data/5-folds/fold${i}/val/ --out ./data/5-folds/fold${i}/val.csv
done