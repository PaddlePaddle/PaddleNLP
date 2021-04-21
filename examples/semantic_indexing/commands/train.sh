#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

train_set_file="${user_train_set_file}"
save_model_path="${user_save_model_path}"

${PYTHON_BIN} -u -m paddle.distributed.launch --gpus "6" \
	train.py \
	--device gpu \
	--save_dir ./checkpoints/${save_model_path} \
	--strategy "ANCE" \
	--batch_size 128 \
	--init_from_ckpt ${init_from_ckpt} \
	--save_steps 2000 \
	--max_seq_length 64 \
	--train_set_file "${train_set_file}" \
	> log/${save_model_path}.log 2>&1
