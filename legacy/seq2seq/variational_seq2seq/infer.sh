#!/bin/bash

set -x
export CUDA_VISIBLE_DEVICES=0
dataset=$1
k=$2
python infer.py \
        --vocab_size 10003 \
        --batch_size 32 \
        --init_scale 0.1 \
        --max_grad_norm 5.0 \
        --dataset_prefix data/${dataset}/${dataset} \
        --use_gpu True \
        --reload_model ${dataset}_model/epoch_${k} \

 
