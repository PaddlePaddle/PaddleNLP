#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

QUESTION=$1

if [ ! -d output ]; then
    mkdir output
fi
if [ ! -d log ]; then
    mkdir log
fi

python3 change_to_rerank.py ${QUESTION}

python3 -u ./src/train_ce.py \
                   --use_cuda true \
                   --verbose true \
                   --do_train false \
                   --do_val false \
                   --do_test true \
                   --batch_size 128 \
                   --init_checkpoint "./checkpoints/ranker" \
                   --test_set "./data/demo.tsv" \
                   --test_save "data/demo.score" \
                   --max_seq_len 384 \
                   --for_cn true \
                   --vocab_path "config/ernie_base_1.0_CN/vocab.txt" \
                   --ernie_config_path "config/ernie_base_1.0_CN/ernie_config.json"
                   1>>log/train.log 2>&1

