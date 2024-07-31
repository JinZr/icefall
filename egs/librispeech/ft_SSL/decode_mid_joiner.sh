# for epoch in $(seq 100 220); do
for epoch in 110 115 120 125 130 135 140 145 150 155 160 165 170; do
    for avg in 40 45 50 55 60 65 70 75 80 85 90; do
    # avg=40
    # epoch=220
        ./hubert_mid_joiner/decode.py \
            --avg $avg \
            --epoch $epoch \
            --max-duration 800 \
            --exp-dir ./hubert_mid_joiner/exp_finetune_epoch291_stop_at_30_epoch/ \
            --final-dim 256 \
            --decoding-method modified_beam_search \
            --beam-size 8
    done
done


for epoch in 130 135 140 145 150 155 160 165 170; do
    for avg in 40 45 50 55 60 65 70 75 80 85 90; do
    # avg=40
    # epoch=220
        CUDA_VISIBLE_DEVICES=4 ./hubert_mid_joiner/decode.py \
            --avg $avg \
            --epoch $epoch \
            --max-duration 800 \
            --exp-dir ./hubert_mid_joiner/exp_finetune_epoch291_stop_at_30_epoch/ \
            --final-dim 256 \
            --decoding-method modified_beam_search \
            --beam-size 8
    done
done

