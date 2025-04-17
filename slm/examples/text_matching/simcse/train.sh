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

python -u -m paddle.distributed.launch --gpus '4' \
	train.py \
	--device gpu \
	--save_dir ./checkpoints/ \
	--batch_size 64 \
	--learning_rate 5E-5 \
	--epochs 8 \
	--save_steps 2000 \
	--eval_steps 100 \
	--max_seq_length 64 \
	--dropout 0.3 \
    --output_emb_size 256 \
    --dup_rate 0.32 \
	--train_set_file "./senteval_cn/STS-B/train.txt" \
	--test_set_file "./senteval_cn/STS-B/dev.tsv" 