for epoch in $(seq 50 60); do
    for avg in $(seq 10 25); do
        ./zipformer_mid_joiner/decode.py \
            --context-size 1 \
            --exp-dir ./zipformer_mid_joiner/exp-0.3-init \
            --avg $avg \
            --epoch $epoch \
            --mid-encoder-dim 384
    done
done

