#!/usr/bin/env bash

for i in 0 1 2 3 4; do
    nohup python3 train.py \
    --fold ${i} \
    --encoder_name se_resnext50_32x4d \
    --batch_size 4 \
    --patience 5 \
    --description se_resnext50_32x4d_batch4_radam_patience_5 \
    --version 28 > ../logs/out_version_28_fold_${i}.log \

    sleep 3;

    python3 predict.py \
    --fold ${i} \
    --encoder_name se_resnext50_32x4d \
    --test_batch_size 8 \
    --version 28
done

###############ENSEMBLE####################
nohup nohup python3 models_ensemble_average.py \
--version 28 > ../logs/out_version_28_ensemble.log


#for i in 0 1 2 3 4; do
#    nohup python3 train.py \
#    --fold ${i} \
#    --encoder_name se_resnext50_32x4d \
#    --batch_size 4 \
#    --patience 5 \
#    --learning_rate 1e-3 \
#    --description lr1e-3_se_resnext50_32x4d_batch4_radam_patience_5 \
#    --version 32 > ../logs/out_version_32_fold_${i}.log \
#
#    sleep 3;
#
#    python3 predict.py \
#    --fold ${i} \
#    --encoder_name se_resnext50_32x4d \
#    --test_batch_size 8 \
#    --version 32
#done
#
################ENSEMBLE####################
#nohup nohup python3 models_ensemble_average.py \
#--version 32 > ../logs/out_version_32_ensemble.log