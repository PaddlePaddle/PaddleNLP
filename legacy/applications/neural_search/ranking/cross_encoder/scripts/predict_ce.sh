#!/bin/bash
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0
python predict.py \
                --device 'gpu' \
                --params_path checkpoints/model_80000/model_state.pdparams \
                --model_name_or_path rocketqa-base-cross-encoder \
                --test_set data/test.csv \
                --topk 10 \
                --batch_size 128 \
                --max_seq_length 384