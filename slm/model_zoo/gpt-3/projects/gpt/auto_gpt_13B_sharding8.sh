#! /bin/bash
# Runs the "1.3B" parameter model
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

log_dir=log_auto
rm -rf $log_dir

# 10B+sharding8 run_pretrain
# Engine.eval_freq in this bash if is set small will cause error (sharding in eval mode has problem)
python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3,4,5,6,7" \
    ./tools/auto.py \
    -c ./ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_13B_sharding8.yaml \
    -o Engine.max_steps=1000 \
    -o Engine.logging_freq=1 \
    -o Engine.eval_freq=10000 \
    -o Engine.verbose=3
