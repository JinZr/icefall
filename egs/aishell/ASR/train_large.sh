./zipformer_mid_joiner/train.py \
    --world-size 2 \
    --num-epochs 60 \
    --use-fp16 1 \
    --context-size 1 \
    --exp-dir ./zipformer_mid_joiner/exp-0.3-init-large/ \
    --enable-musan 0 \
    --base-lr 0.045 \
    --lr-batches 7500 \
    --lr-epochs 18 \
    --spec-aug-time-warp-factor 20 \
    --num-encoder-layers 2,2,4,5,4,2 \
    --feedforward-dim 512,768,1536,2048,1536,768 \
    --encoder-dim 192,256,512,768,512,256 \
    --encoder-unmasked-dim 192,192,256,320,256,192 \
    --max-duration 800 \
    --master-port 4716 \
    --mid-encoder-dim 512 \
    --mid-rnnt-loss-scale 0.3

