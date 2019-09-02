#!/usr/bin/env bash

for i in 0 1 2 3 4; do
    nohup python3 train.py \
    --fold ${i} \
    --encoder_name se_resnext50_32x4d \
    --batch_size 4 \
    --patience 5 \
    --description se_resnext50_32x4d_batch4_radam_patience_5 \
    --version 28 > ../logs/out_version_28_fold_${i}.log \

done

