#!/usr/bin/env bash

for i in 0 1 2 3 4; do
    nohup python3 train.py \
    --fold ${i} \
    --encoder_name se_resnext50_32x4d \
    --batch_size 4 \
    --patience 5 \
    --train_image_path /home/rick/siim_data/stage2/stage2_train \
    --description se_resnext50_32x4d_batch4_radam_patience_5 \
    --version 35 > ../logs/out_version_stage2_fold_${i}_4.log \

    sleep 3;

    python3 predict.py \
    --fold ${i} \
    --encoder_name se_resnext50_32x4d \
    --test_batch_size 8 \
    --stage2_path /home/rick/siim_data/stage2/ \
    --version 35
done

###############ENSEMBLE####################
nohup nohup python3 models_ensemble_average.py \
--version 35 > ../logs/out_ensemble_trained_stage2_4.log



#for i in 0 1 2 3 4; do
#
#    python3 predict.py \
#    --fold ${i} \
#    --encoder_name se_resnext50_32x4d \
#    --test_batch_size 8 \
#    --stage2_path /home/rick/siim_data/stage2/ \
#    --version 28
#done
#
################ENSEMBLE####################
#nohup nohup python3 models_ensemble_average.py \
#--encoder_name se_resnext50_32x4d \
#--version 28 > ../logs/out_stage2_ensemble.log





#for i in 0 1 2 3 4; do
#    nohup python3 train.py \
#    --fold ${i} \
#    --train_image_path /home/rick/siim_data/stage2/stage2_train \
#    --description default_resnext34_32x4d_batch8_radam_patience_3 \
#    --version 33 > ../logs/out_version_stage2_default_fold_${i}.log \
#
#    sleep 3;
#
#    python3 predict.py \
#    --fold ${i} \
#    --stage2_path /home/rick/siim_data/stage2/ \
#    --version 33
#done
#
################ENSEMBLE####################
#nohup nohup python3 models_ensemble_average.py \
#--version 33 > ../logs/out_ensemble_trained_stage2_default.log