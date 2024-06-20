#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

nj=15
stage=-1
stop_stage=100

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/ReazonSpeech
#      You can find FLAC files in this directory.
#      You can download them from https://huggingface.co/datasets/reazon-research/reazonspeech
#
#  - $dl_dir/dataset.json
#      The metadata of the ReazonSpeech dataset.

dl_dir=$PWD/download

. shared/parse_options.sh || exit 1

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
    # This function is from espnet
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "Running prepare.sh"

log "dl_dir: $dl_dir"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Download data"

  # If you have pre-downloaded it to /path/to/ReazonSpeech,
  # you can create a symlink
  #
  #   ln -sfv /path/to/ReazonSpeech $dl_dir/ReazonSpeech
  #
  if [ ! -d $dl_dir/ReazonSpeech/downloads ]; then
    # Download small-v1 by default.
    lhotse download reazonspeech --subset small-v1 $dl_dir
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Prepare ReazonSpeech manifest"
    # We assume that you have downloaded the ReazonSpeech corpus
    # to $dl_dir/ReazonSpeech
    mkdir -p data/manifests
    if [ ! -e data/manifests/.reazonspeech.done ]; then
        lhotse prepare reazonspeech -j $nj $dl_dir/ReazonSpeech data/manifests
        touch data/manifests/.reazonspeech.done
    fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Compute ReazonSpeech fbank"
    if [ ! -e data/manifests/.reazonspeech-validated.done ]; then
        python local/compute_fbank_reazonspeech.py --manifest-dir data/manifests
        python local/validate_manifest.py --manifest data/manifests/reazonspeech_cuts_train.jsonl.gz
        python local/validate_manifest.py --manifest data/manifests/reazonspeech_cuts_dev.jsonl.gz
        python local/validate_manifest.py --manifest data/manifests/reazonspeech_cuts_test.jsonl.gz
        touch data/manifests/.reazonspeech-validated.done
    fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Stage 3: Prepare ReazonSpeech lang_char"
    python local/prepare_lang_char.py data/manifests/reazonspeech_cuts_train.jsonl.gz
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    log "Stage 4: Show manifest statistics"
    python local/display_manifest_statistics.py --manifest-dir data/manifests > data/manifests/manifest_statistics.txt
    cat data/manifests/manifest_statistics.txt
fi