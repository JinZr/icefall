#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

stage=-1
stop_stage=100

dl_dir=$PWD/download

. shared/parse_options.sh || exit 1

vocab_sizes=(
  2000
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

log "Dataset: LibriSpeech"
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Soft link fbank of LibriSpeech"
  mkdir -p data/fbank
  if [ -e ../../librispeech/ASR/data/fbank/.librispeech.done ]; then
    cd data/fbank
    ln -svf $(realpath ../../../../librispeech/ASR/data/fbank/librispeech_cuts*) .
    ln -svf $(realpath ../../../../librispeech/ASR/data/fbank/librispeech_feats*) .
    cd ../..
  else
    log "Abort! Please run ../../librispeech/ASR/prepare.sh --stage 3 --stop-stage 3"
    exit 1
  fi
fi

log "Dataset: AiShell-2"
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Soft link fbank of AiShell-2"
  mkdir -p data/fbank
  if [ -e ../../aishell2/ASR/data/fbank/.aishell2.done ]; then
    cd data/fbank
    ln -svf $(realpath ../../../../aishell2/ASR/data/fbank/aishell2_cuts*) .
    ln -svf $(realpath ../../../../aishell2/ASR/data/fbank/aishell2_feats*) .
    cd ../..
  else
    log "Abort! Please run ../../aishell2/ASR/prepare.sh --stage 3 --stop-stage 3"
    exit 1
  fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Prepare Byte BPE based lang"
  mkdir -p data/fbank
  if [ ! -d ../../aishell2/ASR/data/lang_char ]; then
    log "Abort! Please run ../../aishell2/ASR/prepare.sh --stage 3 --stop-stage 3"
    exit 1
  fi

  if [ ! -d ../../librispeech/ASR/data/lang_phone ]; then
    log "Abort! Please run ../../librispeech/ASR/prepare.sh --stage 5 --stop-stage 5"
    exit 1
  fi

  if [ ! -d ../../librispeech/ASR/data/lang_bpe_500 ]; then
    log "Abort! Please run ../../librispeech/ASR/prepare.sh --stage 6 --stop-stage 6"
    exit 1
  fi

  cd data/
  ln -svf $(realpath ../../../aishell2/ASR/data/lang_char) .
  ln -svf $(realpath ../../../librispeech/ASR/data/lang_phone) .
  ln -svf $(realpath ../../../librispeech/ASR/data/lang_bpe_500) .
  cd ../

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bbpe_${vocab_size}
    mkdir -p $lang_dir

    cat data/lang_char/text data/lang_bpe_500/transcript_words.txt \
      > $lang_dir/text

    ./local/prepare_for_bpe_model.py \
      --lang_dir ./$lang_dir \
      --text $lang_dir/text
    
    if [ ! -f $lang_dir/text_words_segmentation ]; then
      python3 ./local/text2segments.py \
        --input-file ./data/lang_char/text \
        --output-file $lang_dir/text_words_segmentation
      
      cat ./data/lang_bpe_500/transcript_words.txt \
        >> $lang_dir/text_words_segmentation
    fi

    cat $lang_dir/text_words_segmentation | sed 's/ /\n/g' \
      | sort -u | sed '/^$/d' | uniq > $lang_dir/words_no_ids.txt

    if [ ! -f $lang_dir/words.txt ]; then
      python3 ./local/prepare_words.py \
        --input-file $lang_dir/words_no_ids.txt \
        --output-file $lang_dir/words.txt
    fi

    if [ ! -f $lang_dir/bbpe.model ]; then
      ./local/train_bbpe_model.py \
        --lang-dir $lang_dir \
        --vocab-size $vocab_size \
        --transcript $lang_dir/transcript_chars.txt
    fi

    if [ ! -f $lang_dir/L_disambig.pt ]; then
      ./local/prepare_lang_bbpe.py --lang-dir $lang_dir

      log "Validating $lang_dir/lexicon.txt"
      ./local/validate_bpe_lexicon.py \
        --lexicon $lang_dir/lexicon.txt \
        --bpe-model $lang_dir/bbpe.model
    fi
  done 
fi

