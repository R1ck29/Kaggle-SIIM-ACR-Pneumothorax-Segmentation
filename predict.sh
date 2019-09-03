#!/usr/bin/env bash

for i in 0 1 2 3 4; do

    python3 predict.py \
    --fold ${i} \
    --encoder_name se_resnext50_32x4d \
    --test_batch_size 8 \
    --version 28 \
    --stage2_path /home/rick/siim_data/stage2/ \
    --best_score_weight True
done

###############ENSEMBLE####################
nohup nohup python3 models_ensemble_average.py \
--best_score_weight True \
--version 28 > ./logs/out_version_stage2_ensemble_best_score.log