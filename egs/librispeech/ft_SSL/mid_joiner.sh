./hubert_mid_joiner/finetune.py   \
    --world-size 8   \
    --num-epochs 222   \
    --start-epoch 88   \
    --use-fp16 1    \
    --exp-dir hubert_mid_joiner/exp_finetune_epoch291_stop_at_40_epoch   \
    --pretrained-dir pretrained/hubert_base_ls960.pt  \
    --full-libri 0  \
    --max-duration 200  \
    --accum-grad 1  \
    --do-normalize 0   \
    --final-dim 256   \
    --mask-prob 0.65  \
    --mask-channel-prob 0.5 \
    --mask-channel-length 64  \
    --encoder-layerdrop 0.1  \
    --activation-dropout 0.1  \
    --feature-grad-mult 0.0   \
    --base-lr 0.001 \
    --num-workers 1
