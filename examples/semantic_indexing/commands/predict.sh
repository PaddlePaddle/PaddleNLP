#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

if [[ $# != 1 ]]; then
	echo "Usage: Bash $0 input_file"
	exit 1
fi

input_file=$1

${PYTHON_BIN} -u -m paddle.distributed.launch --gpus "0" \
	predict.py \
	--device gpu \
	--params_path ${pretrained_model_params_path} \
	--batch_size 128 \
	--max_seq_length 64 \
	--input_file ${input_file}
