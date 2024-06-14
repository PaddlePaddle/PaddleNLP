#!/bin/bash

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

DATASET=$1

if [ "$DATASET" == cnndm ]
then
python -m paddle.distributed.launch --gpus 0 python train_prophetnet.py \
    --dataset=cnndm \
    --model_name_or_path=prophetnet-large-uncased \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=8 \
    --num_train_epochs=4 \
    --learning_rate=0.0001 \
    --warmup_init_lr=1e-07 \
    --warmup_steps=1000 \
    --max_grad_norm=0.1 \
    --dataloader_num_workers=4 \
    --logging_steps 10 \
    --save_steps 100 \
    --do_train \
    --do_eval \
    --output_dir=./ckpt/cnndm
else
python -m paddle.distributed.launch --gpus 0 python train_prophetnet.py \
    --dataset=gigaword \
    --model_name_or_path=prophetnet-large-uncased \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=32 \
    --num_train_epochs=6 \
    --learning_rate=0.0001 \
    --warmup_init_lr=1e-07 \
    --warmup_steps=1000 \
    --max_grad_norm=0.1 \
    --dataloader_num_workers=8 \
    --logging_steps 10 \
    --save_steps 100 \
    --do_train \
    --do_eval \
    --output_dir=./ckpt/gigaword
fi