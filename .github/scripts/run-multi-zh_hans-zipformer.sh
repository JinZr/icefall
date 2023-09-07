#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/multi_zh-hans/ASR

repo_url=https://huggingface.co/zrjin/icefall-asr-multi-zh-hans-zipformer-2023-9-2/

log "Downloading pre-trained model from $repo_url"
git lfs install
git clone $repo_url
repo=$(basename $repo_url)


log "Display test files"
tree $repo/
ls -lh $repo/test_wavs/*.wav

pushd $repo/exp
ln -s epoch-20.pt epoch-99.pt
popd

log "Test exporting to ONNX format"

./zipformer/export.py \
  --exp-dir $repo/exp \
  --tokens $repo/data/lang_bpe_2000/tokens.txt \
  --epoch 99 \
  --avg 1 \
  --onnx 1

log "Export to torchscript model"

./zipformer/export.py \
  --exp-dir $repo/exp \
  --tokens $repo/data/lang_bpe_2000/tokens.txt \
  --epoch 99 \
  --avg 1 \
  --jit 1

./zipformer/export.py \
  --exp-dir $repo/exp \
  --tokens $repo/data/lang_bpe_2000/tokens.txt \
  --epoch 99 \
  --avg 1 \
  --jit-trace 1

ls -lh $repo/exp/*.onnx
ls -lh $repo/exp/*.pt

log "Decode with ONNX models"

./zipformer/onnx_check.py \
  --jit-filename $repo/exp/cpu_jit.pt \
  --onnx-encoder-filename $repo/exp/encoder.onnx \
  --onnx-decoder-filename $repo/exp/decoder.onnx \
  --onnx-joiner-filename $repo/exp/joiner.onnx \
  --onnx-joiner-encoder-proj-filename $repo/exp/joiner_encoder_proj.onnx \
  --onnx-joiner-decoder-proj-filename $repo/exp/joiner_decoder_proj.onnx

./zipformer/onnx_pretrained.py \
  --tokens $repo/data/lang_bpe_2000/tokens.txt \
  --encoder-model-filename $repo/exp/encoder.onnx \
  --decoder-model-filename $repo/exp/decoder.onnx \
  --joiner-model-filename $repo/exp/joiner.onnx \
  --joiner-encoder-proj-model-filename $repo/exp/joiner_encoder_proj.onnx \
  --joiner-decoder-proj-model-filename $repo/exp/joiner_decoder_proj.onnx \
  $repo/test_wavs/DEV_T0000000000.wav \
  $repo/test_wavs/DEV_T0000000001.wav \
  $repo/test_wavs/DEV_T0000000002.wav

log "Decode with models exported by torch.jit.trace()"

./zipformer/jit_pretrained.py \
  --tokens $repo/data/lang_bpe_2000/tokens.txt \
  --encoder-model-filename $repo/exp/encoder_jit_trace.pt \
  --decoder-model-filename $repo/exp/decoder_jit_trace.pt \
  --joiner-model-filename $repo/exp/joiner_jit_trace.pt \
  $repo/test_wavs/DEV_T0000000000.wav \
  $repo/test_wavs/DEV_T0000000001.wav \
  $repo/test_wavs/DEV_T0000000002.wav

./zipformer/jit_pretrained.py \
  --tokens $repo/data/lang_bpe_2000/tokens.txt \
  --encoder-model-filename $repo/exp/encoder_jit_script.pt \
  --decoder-model-filename $repo/exp/decoder_jit_script.pt \
  --joiner-model-filename $repo/exp/joiner_jit_script.pt \
  $repo/test_wavs/DEV_T0000000000.wav \
  $repo/test_wavs/DEV_T0000000001.wav \
  $repo/test_wavs/DEV_T0000000002.wav

for sym in 1 2 3; do
  log "Greedy search with --max-sym-per-frame $sym"

  ./zipformer/pretrained.py \
    --checkpoint $repo/exp/epoch-99.pt \
    --lang-dir $repo/data/lang_bpe_2000 \
    --decoding-method greedy_search \
    --max-sym-per-frame $sym \
  $repo/test_wavs/DEV_T0000000000.wav \
  $repo/test_wavs/DEV_T0000000001.wav \
  $repo/test_wavs/DEV_T0000000002.wav
done

for method in modified_beam_search beam_search fast_beam_search; do
  log "$method"

  ./zipformer/pretrained.py \
    --decoding-method $method \
    --beam-size 4 \
    --checkpoint $repo/exp/epoch-99.pt \
    --lang-dir $repo/data/lang_bpe_2000 \
  $repo/test_wavs/DEV_T0000000000.wav \
  $repo/test_wavs/DEV_T0000000001.wav \
  $repo/test_wavs/DEV_T0000000002.wav
done
