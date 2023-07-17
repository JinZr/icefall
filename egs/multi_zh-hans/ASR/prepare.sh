#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

nj=16
stage=-1
stop_stage=100

dl_dir=$PWD/download

. shared/parse_options.sh || exit 1

vocab_sizes=(
  # 2000
  # 1000
  500
)


# multidataset list.
# LibriSpeech and musan are required.
# The others are optional.
multidataset=(
  "gigaspeech",
  "commonvoice",
  "librilight",
)

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

log "Dataset: musan"
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Soft link fbank of musan"
  mkdir -p data/fbank
  if [ -e ../../librispeech/ASR/data/fbank/.musan.done ]; then
    cd data/fbank
    ln -svf $(realpath ../../../../librispeech/ASR/data/fbank/musan_feats) .
    ln -svf $(realpath ../../../../librispeech/ASR/data/fbank/musan_cuts.jsonl.gz) .
    cd ../..
  else
    log "Abort! Please run ../../librispeech/ASR/prepare.sh --stage 4 --stop-stage 4"
    exit 1
  fi
fi

log "Dataset: THCHS-30"
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Prepare THCHS-30"
  if [ ! -d $dl_dir/thchs30 ]; then
    log "Downloading THCHS-30"
    lhotse download thchs30 $dl_dir/thchs30
  fi

  if [ ! -f data/manifests/.thchs30.done ]; then
    mkdir -p data/manifests
    lhotse prepare thchs-30 $dl_dir/thchs30 data/manifests/thchs30
    touch data/manifests/.thchs30.done
  fi

  if [ ! -f data/fbank/.thchs30.done ]; then
    mkdir -p data/fbank
    ./local/compute_fbank_thchs30.py
    touch data/fbank/.thchs30.done
  fi
fi

log "Dataset: AISHELL-1"
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Prepare AISHELL-1"
  if [ -e ../../aishell/ASR/data/fbank/.aishell.done ]; then
    cd data/fbank
    ln -svf $(realpath ../../../../aishell/ASR/data/fbank/aishell_feats_train) .
    ln -svf $(realpath ../../../../aishell/ASR/data/fbank/aishell_feats_dev) .
    ln -svf $(realpath ../../../../aishell/ASR/data/fbank/aishell_feats_test) .
    ln -svf $(realpath ../../../../aishell/ASR/data/fbank/aishell_cuts_train.jsonl.gz) .
    ln -svf $(realpath ../../../../aishell/ASR/data/fbank/aishell_cuts_dev.jsonl.gz) .
    ln -svf $(realpath ../../../../aishell/ASR/data/fbank/aishell_cuts_test.jsonl.gz) .
    cd ../..
  else
    log "Abort! Please run ../../aishell/ASR/prepare.sh --stage 3 --stop-stage 3"
    exit 1
  fi
fi

log "Dataset: AISHELL-2"
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Prepare AISHELL-2"
  if [ -e ../../aishell/ASR/data/fbank/.aishell2.done ]; then
    cd data/fbank
    ln -svf $(realpath ../../../../aishell2/ASR/data/fbank/aishell2_feats_train) .
    ln -svf $(realpath ../../../../aishell2/ASR/data/fbank/aishell_feats_dev) .
    ln -svf $(realpath ../../../../aishell2/ASR/data/fbank/aishell_feats_test) .
    ln -svf $(realpath ../../../../aishell2/ASR/data/fbank/aishell_cuts_train.jsonl.gz) .
    ln -svf $(realpath ../../../../aishell2/ASR/data/fbank/aishell_cuts_dev.jsonl.gz) .
    ln -svf $(realpath ../../../../aishell2/ASR/data/fbank/aishell_cuts_test.jsonl.gz) .
    cd ../..
  else 
    log "Abort! Please run ../../aishell2/ASR/prepare.sh --stage 3 --stop-stage 3"
    exit 1
  fi 
fi