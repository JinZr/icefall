./zipformer_mid_output_final_joiner/train.py \
    --world-size 4 \
    --num-epochs 40 \
    --use-fp16 1 \
    --exp-dir ./zipformer_mid_output_final_joiner/exp-0.3-full-downsample-large \
    --num-encoder-layers 2,2,4,5,4,2 \
    --feedforward-dim 512,768,1536,2048,1536,768 \
    --encoder-dim 192,256,512,768,512,256 \
    --encoder-unmasked-dim 192,192,256,320,256,192 \
    --max-duration 1000 \
    --mid-rnnt-loss-scale 0.3 \
    --master-port 2414 \
    --mid-encoder-dim 512 

