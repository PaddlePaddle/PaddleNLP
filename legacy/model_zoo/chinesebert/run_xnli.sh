# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

python -m paddle.distributed.launch --gpus 0,1 python train_xnli.py \
--data_path './data/XNLI' \
--device 'gpu' \
--num_train_epochs 5 \
--max_seq_length 256 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--learning_rate 1.3e-5 \
--adam_beta2 0.98 \
--weight_decay 0.001 \
--warmup_ratio 0.1 \
--logging_steps 10 \
--save_steps 100 \
--seed 2333 \
--do_train \
--do_eval \
--output_dir "outputs/xnli" | tee outputs/train_xnli.log
