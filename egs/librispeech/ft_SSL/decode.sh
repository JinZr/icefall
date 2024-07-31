for epoch in $(seq 170 220); do
    for avg in $(seq 70 120); do
        ./hubert/decode.py \
            --avg $avg \
            --epoch $epoch \
            --max-duration 800 \
            --exp-dir ./hubert/exp_finetune_epoch291/ \
            --final-dim 256 # \
            # --decoding-method modified_beam_search \
            # --beam-size 8
    done
done