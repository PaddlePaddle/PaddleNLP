#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=0
dataset=$1
python train.py \
        --vocab_size 10003 \
        --batch_size 32 \
        --init_scale 0.1 \
        --max_grad_norm 5.0 \
        --dataset_prefix data/${dataset}/${dataset} \
        --model_path ${dataset}_model\
        --use_gpu True \
        --max_epoch 200 \

 
