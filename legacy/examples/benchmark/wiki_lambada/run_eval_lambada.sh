#!/bin/bash

if [[ ! -e "lambada_test.jsonl" ]]; then
    wget https://paddlenlp.bj.bcebos.com/data/benchmark/lambada_test.jsonl
fi

python3.10 eval.py \
--model_name_or_path /home/tianyu.zhou/ZIBO4/1005/checkpoint-all-nodes \
--batch_size 4 \
--eval_path lambada_test.jsonl \
--tensor_parallel_degree 1 \
--cloze_eval \
--use_flash_attention true
