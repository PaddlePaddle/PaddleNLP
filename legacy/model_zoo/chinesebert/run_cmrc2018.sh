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

python -m paddle.distributed.launch --gpus 0,1 python train_cmrc2018.py \
    --data_dir "./data/cmrc2018" \
    --model_name_or_path ChineseBERT-large \
    --max_seq_length 512 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --learning_rate 4e-5 \
    --max_grad_norm 1.0 \
    --adam_beta2 0.98 \
    --num_train_epochs 3 \
    --logging_steps 2 \
    --save_steps 20 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --seed 1111 \
    --do_train \
    --do_eval \
    --dataloader_num_workers 0 \
    --fp16 True \
    --output_dir "outputs/cmrc2018"

