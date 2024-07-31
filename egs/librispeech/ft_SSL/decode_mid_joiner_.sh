./hubert_mid_joiner/decode.py \
    --avg 60 \
    --epoch 160 \
    --max-duration 800 \
    --exp-dir ./hubert_mid_joiner/exp_finetune_epoch291_stop_at_30_epoch/ \
    --final-dim 256 \
    --decoding-method modified_beam_search \
    --beam-size 8

