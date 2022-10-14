#!/bin/bash

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

mkdir -p output
unset CUDA_VISIBLE_DEVICES
TRAIN_SET="../dureader-retrieval-baseline-dataset/train/cross.train.tsv"
node=4
epoch=3
lr=1e-5
batch_size=32
train_exampls=`cat $TRAIN_SET | wc -l`
save_steps=$[$train_exampls/$batch_size/$node]
new_save_steps=$[$save_steps*$epoch/2]

python -u -m paddle.distributed.launch --gpus "0,1,2,3" train_ce.py \
        --device gpu \
        --train_set ${TRAIN_SET} \
        --save_dir ./checkpoints \
        --batch_size ${batch_size} \
        --save_steps ${new_save_steps} \
        --max_seq_len 384 \
        --learning_rate 1E-5 \
        --weight_decay  0.01 \
        --warmup_proportion 0.0 \
        --logging_steps 10 \
        --seed 1 \
        --epochs ${epoch}