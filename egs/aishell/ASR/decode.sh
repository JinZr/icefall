for epoch in $(seq 55 55); do
    for avg in $(seq 19 19); do
        ./zipformer_mid_joiner/decode.py \
            --context-size 1 \
            --exp-dir ./zipformer_mid_joiner/exp-0.3-init \
            --avg $avg \
            --epoch $epoch \
            --mid-encoder-dim 384 \
            --decoding-method modified_beam_search
    done
done

