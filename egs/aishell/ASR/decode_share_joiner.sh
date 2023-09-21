for epoch in $(seq 47 60); do
    for avg in $(seq 5 20); do
        ./zipformer_mid_output_final_joiner/decode.py \
            --context-size 1 \
            --exp-dir ./zipformer_mid_output_final_joiner/exp-0.3-init \
            --mid-rnnt-loss-scale 0.3 \
            --mid-encoder-dim 384 \
            --avg $avg \
            --epoch $epoch
    done
done
