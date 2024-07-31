for epoch in $(seq 20 50); do
    for avg in $(seq 20 40); do
        ./hubert_mid_joiner/decode.py \
            --avg ${avg} \
            --epoch ${epoch} \
            --exp-dir ./hubert_mid_joiner/large_exp_finetune_epoch291_stop_at_30_epoch/ \
            --extractor-mode "layer_norm" \
            --encoder-attention-heads 16 \
            --untie-final-proj 1 \
            --encoder-layers 24 \
            --do-normalize 0   \
            --final-dim 768   \
            --encoder-embed-dim 1024 \
            --encoder-ffn-embed-dim 4096 \
            --layer-norm-first 1 \
            --mid-encoder-dim 1024 
    done
done
