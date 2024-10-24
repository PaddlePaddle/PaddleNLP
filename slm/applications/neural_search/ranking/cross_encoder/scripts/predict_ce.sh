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
export CUDA_VISIBLE_DEVICES=0
python predict.py \
                --device 'gpu' \
                --params_path checkpoints/model_80000/model_state.pdparams \
                --model_name_or_path rocketqa-base-cross-encoder \
                --test_set data/test.csv \
                --topk 10 \
                --batch_size 128 \
                --max_seq_length 384