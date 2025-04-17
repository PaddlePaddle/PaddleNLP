# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# simcse gpu
python -u -m paddle.distributed.launch --gpus '1,2,3,4' \
	train.py \
	--device gpu \
	--save_dir ./checkpoints/ \
	--batch_size 64 \
	--learning_rate 5E-5 \
	--epochs 3 \
	--save_steps 2000 \
	--eval_steps 100 \
	--max_seq_length 64 \
	--infer_with_fc_pooler \
	--dropout 0.2 \
	--output_emb_size 256 \
	--train_set_file "./recall/train_unsupervised.csv" \
	--test_set_file "./recall/dev.csv" \
	--model_name_or_path "rocketqa-zh-base-query-encoder"

# simcse cpu
# python 	train.py \
# 	--device cpu \
# 	--save_dir ./checkpoints/ \
# 	--batch_size 64 \
# 	--learning_rate 5E-5 \
# 	--epochs 3 \
# 	--save_steps 2000 \
# 	--eval_steps 100 \
# 	--max_seq_length 64 \
# 	--infer_with_fc_pooler \
# 	--dropout 0.2 \
#	--output_emb_size 256 \
# 	--train_set_file "./recall/train_unsupervised.csv" \
# 	--test_set_file "./recall/dev.csv" 
# 	--model_name_or_path "ernie-3.0-medium-zh"

# post training + simcse
# python -u -m paddle.distributed.launch --gpus '0,1,2,3' \
# 	train.py \
# 	--device gpu \
# 	--save_dir ./checkpoints/ \
# 	--batch_size 64 \
# 	--learning_rate 5E-5 \
# 	--epochs 3 \
# 	--save_steps 2000 \
# 	--eval_steps 100 \
# 	--max_seq_length 64 \
# 	--infer_with_fc_pooler \
# 	--dropout 0.2 \
#	--output_emb_size 256 \
# 	--train_set_file "./recall/train_unsupervised.csv" \
# 	--test_set_file "./recall/dev.csv" 
# 	--model_name_or_path "post_ernie"



