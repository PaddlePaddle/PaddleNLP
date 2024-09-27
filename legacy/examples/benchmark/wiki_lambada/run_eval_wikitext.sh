#!/bin/bash

if [[ ! -d "wikitext-103" ]]; then
    wget https://paddlenlp.bj.bcebos.com/data/benchmark/wikitext-103.tar.gz
    tar -zvf wikitext-103.tar.gz
fi

python3.10 eval.py \
--model_name_or_path /home/tianyu.zhou/ZIBO4/1005/checkpoint-all-nodes \
--batch_size 4 \
--eval_path wikitext-103/wiki.valid.tokens \
--tensor_parallel_degree 1 \
--use_flash_attention true
