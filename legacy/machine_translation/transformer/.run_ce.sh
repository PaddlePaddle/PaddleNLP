#!/bin/bash

DATA_PATH=./dataset/wmt16

train(){
    python -u main.py \
        --do_train True \
        --src_vocab_fpath $DATA_PATH/en_10000.dict \
        --trg_vocab_fpath $DATA_PATH/de_10000.dict \
        --special_token '<s>' '<e>' '<unk>' \
        --training_file $DATA_PATH/wmt16/train \
        --use_token_batch True \
        --batch_size 2048 \
        --sort_type pool \
        --pool_size 10000 \
        --print_step 1 \
        --weight_sharing False \
        --epoch 20 \
        --enable_ce True \
        --random_seed 1000 \
        --save_checkpoint "" \
        --save_param ""
}

cudaid=${transformer:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

train | python _ce.py

cudaid=${transformer_m:=0,1,2,3} # use 0,1,2,3 card as default
export CUDA_VISIBLE_DEVICES=$cudaid

train | python _ce.py