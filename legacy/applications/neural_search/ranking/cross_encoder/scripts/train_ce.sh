#!/bin/bash
unset CUDA_VISIBLE_DEVICES
python -u -m paddle.distributed.launch --gpus "0,1,2,3" --log_dir="logs" train_ce.py \
        --device gpu \
        --train_set data/train.csv \
        --test_file data/dev_pairwise.csv \
        --save_dir ./checkpoints \
        --model_name_or_path rocketqa-base-cross-encoder \
        --batch_size 32 \
        --save_steps 10000 \
        --max_seq_len 384 \
        --learning_rate 1E-5 \
        --weight_decay  0.01 \
        --warmup_proportion 0.0 \
        --logging_steps 10 \
        --seed 1 \
        --epochs 3 \
        --eval_step 1000
