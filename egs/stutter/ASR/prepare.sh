#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

nj=15
stage=-1
stop_stage=11
perturb_speed=true

# Aishell-1 corpus is allowed to be used in this challenge
use_aishell=false

dl_dir=$PWD/download

# We assume you have used the official script 
# https://github.com/hongfeixue/StutteringSpeechChallenge
# to process the data and put kaldi format data in $data_dir
data_dir=$PWD/download

. shared/parse_options.sh || exit 1

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "stage 0: Download data"

  # If you have pre-downloaded it to /path/to/aishell,
  # you can create a symlink
  #
  #   ln -sfv /path/to/aishell $dl_dir/aishell
  #
  # The directory structure is
  # aishell/
  # |-- data_aishell
  # |   |-- transcript
  # |   `-- wav
  # `-- resource_aishell
  #     |-- lexicon.txt
  #     `-- speaker.info

  if [ $use_aishell = true ] && [ ! -d $dl_dir/aishell/data_aishell/wav/train ]; then
    lhotse download aishell $dl_dir
  fi

  # If you have pre-downloaded it to /path/to/musan,
  # you can create a symlink
  #
  #   ln -sfv /path/to/musan $dl_dir/musan
  #
  if [ ! -d $dl_dir/musan ]; then
    lhotse download musan $dl_dir
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare Stuttering Speech Challenge manifest"
  # We assume that you have downloaded the aishell corpus
  # to $dl_dir/aishell
  if [ ! -f data/manifests/.stutter_manifests.done ]; then
    mkdir -p data/manifests

    if [ ! -f ${data_dir}/.stutter_preprocess.done ]; then
      for dir in train dev test; do
          if [ ! -d ${data_dir}/$dir ]; then
              log "Error: ${data_dir}/$dir does not exist"
              exit 1
          else
              ./local/preprocess_stutter.py --kaldi-dir ${data_dir}/$dir 
              mv ${data_dir}/$dir/text ${data_dir}/$dir/text.orig
              mv ${data_dir}/$dir/text.preprocessed ${data_dir}/$dir/text
          fi
      done

      touch ${data_dir}/.stutter_preprocess.done
    fi

    if [ ! -f ${data_dir}/.stutter_lhotse.done ]; then
      for dir in train dev test; do
          if [ ! -d ${data_dir}/$dir ]; then
              log "Error: ${data_dir}/$dir does not exist"
              exit 1
          else
            lhotse kaldi import ${data_dir}/$dir 16000 data/manifests/

            mv data/manifests/cuts.jsonl.gz data/manifests/stutter_cuts_${dir}.jsonl.gz
            mv data/manifests/recordings.jsonl.gz data/manifests/stutter_recordings_${dir}.jsonl.gz
            mv data/manifests/supervisions.jsonl.gz data/manifests/stutter_supervisions_${dir}.jsonl.gz
          fi
      done

      touch ${data_dir}/.stutter_lhotse.done
    fi

    touch data/manifests/.stutter_manifests.done
  fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Prepare aishell manifest"
  # We assume that you have downloaded the aishell corpus
  # to $dl_dir/aishell
  if [ $use_aishell = true ] && [ ! -f data/manifests/.aishell_manifests.done ]; then
    mkdir -p data/manifests
    lhotse prepare aishell $dl_dir/aishell data/manifests
    touch data/manifests/.aishell_manifests.done
  else
    log "Skip stage 2: Prepare aishell manifest"
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Prepare musan manifest"
  # We assume that you have downloaded the musan corpus
  # to data/musan
  if [ ! -f data/manifests/.musan_manifests.done ]; then
    log "It may take 6 minutes"
    mkdir -p data/manifests
    lhotse prepare musan $dl_dir/musan data/manifests
    touch data/manifests/.musan_manifests.done
  fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Compute fbank for aishell"
  if [ ! -f data/fbank/.aishell.done ]; then
    mkdir -p data/fbank
    ./local/compute_fbank_aishell.py --perturb-speed ${perturb_speed}
    touch data/fbank/.aishell.done
  fi
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Compute fbank for Stuttering Speech Challenge"
  if [ ! -f data/fbank/.stutter.done ]; then
    mkdir -p data/fbank
    ./local/compute_fbank_stutter.py --perturb-speed ${perturb_speed}
    touch data/fbank/.stutter.done
  fi
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Compute fbank for musan"
  if [ ! -f data/fbank/.msuan.done ]; then
    mkdir -p data/fbank
    ./local/compute_fbank_musan.py
    touch data/fbank/.msuan.done
  fi
fi

lang_char_dir=data/lang_char
if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  log "Stage 7: Prepare char based lang"
  mkdir -p $lang_char_dir

  if [ ! -f $lang_char_dir/text ]; then
    cat ${data_dir}/train/text > $lang_char_dir/text.orig
    # cat ${data_dir}/dev/text >> $lang_char_dir/text
    # cat ${data_dir}/test/text >> $lang_char_dir/text
  fi

  if [ ! -f $lang_char_dir/text.segment ]; then
    cat ${data_dir}/train/text.segment > $lang_char_dir/text.segment
    # cat ${data_dir}/dev/text.segment >> $lang_char_dir/text.segment
    # cat ${data_dir}/test/text.segment >> $lang_char_dir/text.segment
  fi

  (echo '<eps> 0'; echo '!SIL 1'; echo '<SPOKEN_NOISE> 2'; echo '<UNK> 3';) \
    > $lang_char_dir/words.txt

  cat $lang_char_dir/text.orig | cut -d " " -f 2- > $lang_char_dir/text

  cat $lang_char_dir/text.segment |  cut -d " " -f 2- | sed 's/ /\n/g' | sort -u | sed '/^$/d' \
     | awk '{print $1" "NR+3}' >> $lang_char_dir/words.txt

  num_lines=$(< $lang_char_dir/words.txt wc -l)
  (echo "#0 $num_lines"; echo "<s> $(($num_lines + 1))"; echo "</s> $(($num_lines + 2))";) \
    >> $lang_char_dir/words.txt

  if [ ! -f $lang_char_dir/L_disambig.pt ]; then
    ./local/prepare_char.py --lang-dir $lang_char_dir
  fi
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Prepare G"

  mkdir -p data/lm

  # Train LM on transcripts
  if [ ! -f data/lm/3-gram.unpruned.arpa ]; then
    cat $lang_char_dir/text.segment |  cut -d " " -f 2- \
      > $lang_char_dir/text.lm

    python3 ./shared/make_kn_lm.py \
      -ngram-order 3 \
      -text $lang_char_dir/text.lm \
      -lm data/lm/3-gram.unpruned.arpa
  fi

  # We assume you have installed kaldilm, if not, please install
  # it using: pip install kaldilm
  if [ ! -f data/lm/G_3_gram_char.fst.txt ]; then
    # It is used in building HLG

    python3 -m kaldilm \
      --read-symbol-table="$lang_char_dir/words.txt" \
      --disambig-symbol='#0' \
      --max-order=3 \
      data/lm/3-gram.unpruned.arpa > data/lm/G_3_gram_char.fst.txt
  fi

  if [ ! -f $lang_char_dir/HLG.fst ]; then
    ./local/prepare_lang_fst.py  \
      --lang-dir $lang_char_dir \
      --ngram-G ./data/lm/G_3_gram_char.fst.txt
  fi
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  log "Stage 7: Compile LG & HLG"
  
  ./local/compile_hlg.py --lang-dir $lang_char_dir --lm G_3_gram_char

  ./local/compile_lg.py --lang-dir $lang_char_dir --lm G_3_gram_char

fi