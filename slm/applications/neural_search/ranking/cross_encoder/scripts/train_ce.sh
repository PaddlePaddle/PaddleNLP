#!/bin/bash

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

unset CUDA_VISIBLE_DEVICES
python -u -m paddle.distributed.launch --gpus "0,1,2,3" --log_dir="logs" train_ce.py \
        --device gpu \
        --train_set data/train.csv \
        --test_file data/dev_pairwise.csv \
        --save_dir ./checkpoints \
        --model_name_or_path rocketqa-base-cross-encoder \
        --batch_size 32 \
        --save_steps 10000 \
        --max_seq_len 384 \
        --learning_rate 1E-5 \
        --weight_decay  0.01 \
        --warmup_proportion 0.0 \
        --logging_steps 10 \
        --seed 1 \
        --epochs 3 \
        --eval_step 1000
