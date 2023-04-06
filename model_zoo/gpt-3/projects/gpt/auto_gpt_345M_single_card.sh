#! /bin/bash

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

# log_dir=log_auto_fused_linear_pass_single_card_2
# log_dir=log_auto_fused_linear_mp8
log_dir=log_auto_linear_single_card_2
# log_dir=log_auto_no_fused_linear_single_card
rm -rf $log_dir

# NSYS_CMD="nsys profile -o paddle_clip_perf --stats true --force-overwrite true"

# export FLAGS_USE_STANDALONE_EXECUTOR=False
# export CUDA_VISIBLE_DEVICES=0
# python ./tools/auto.py -c ./ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml 
$NSYS_CMD python -m paddle.distributed.launch --log_dir $log_dir --devices "5" \
    ./tools/auto.py \
    -c ./ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
    -o Engine.verbose=3 \
    -o Engine.logging_freq=1
