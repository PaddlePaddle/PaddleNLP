#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

corpus="${user_corpus_file}"
similar_text_pair="${user_similar_text_pair}"
recall_num=50
batch_size=128
max_len=64
hnsw_ef=100
hnsw_m=100

function run_ann() {
	${PYTHON_BIN} -u -m paddle.distributed.launch --gpus "5" \
		run_ann.py \
		--device gpu \
		--recall_result ${recall_result} \
		--params_path ${init_checkpoint} \
		--hnsw_m ${hnsw_m} \
		--hnsw_ef ${hnsw_ef} \
		--batch_size ${batch_size} \
		--max_seq_length ${max_len} \
		--recall_num ${recall_num} \
		--similar_text_pair ${similar_text_pair} \
		--corpus_file ${corpus} 1>recall_log/${recall_result} 2>&1
}

function evaluate() {
	${PYTHON_BIN} -u evaluate.py \
		--similar_pair_file ${similar_text_pair} \
		--recall_result ${recall_result} \
		--recall_num ${recall_num} > evaluate_log/${recall_result} 2>&1
}

run_ann
evaluate
