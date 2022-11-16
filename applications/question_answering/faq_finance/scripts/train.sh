# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

python -u -m paddle.distributed.launch --gpus='1' \
	train.py \
	--device gpu \
	--model_name_or_path rocketqa-zh-base-query-encoder \
	--save_dir ./checkpoints/ \
	--batch_size 64 \
	--learning_rate 5E-5 \
	--epochs 3 \
	--save_steps 50 \
	--eval_steps 50 \
	--max_seq_length 64 \
	--dropout 0.2 \
	--output_emb_size 256 \
	--dup_rate 0.1 \
	--rdrop_coef 0.1 \
	--train_set_file "./data/train_aug.csv" 