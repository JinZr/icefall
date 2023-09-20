./zipformer_mid_joiner/decode.py \
    --exp-dir ./zipformer_mid_joiner/exp-0.3-full-downsample-large/ \
    --num-encoder-layers 2,2,4,5,4,2 \
    --feedforward-dim 512,768,1536,2048,1536,768 \
    --encoder-dim 192,256,512,768,512,256 \
    --encoder-unmasked-dim 192,192,256,320,256,192 \
    --mid-encoder-dim 512 \
    --epoch 26 \
    --avg 7
