#! /bin/bash

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

export FLAGS_new_executor_micro_batching=True
export FLAGS_enable_pir_in_executor=1
export FLAGS_enable_prim_after_distribute=True
# export GLOG_v=4

log_dir=log_newir
rm -rf $log_dir
rm -rf program_*
rm -rf debug_program_*

# Only dp_degree, mp_degree and pp_degree can be changed
# Only support pure dp, pure mp and pure pp
export CUDA_VISIBLE_DEVICES=5,6
python -m paddle.distributed.launch --log_dir $log_dir --devices "5,6" \
    ./tools/auto.py \
    -c ./ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
    -o Model.num_layers=2 \
    -o Model.hidden_dropout_prob=0 \
    -o Model.attention_probs_dropout_prob=0 \
    -o Model.use_recompute=False \
    -o Global.local_batch_size=2 \
    -o Global.micro_batch_size=1 \
    -o Distributed.dp_degree=1 \
    -o Distributed.mp_degree=1 \
    -o Distributed.pp_degree=2 \
    -o Distributed.sharding.sharding_degree=1 \
    -o Distributed.sharding.sharding_stage=1 \
    -o Engine.max_steps=2 \
    -o Engine.eval_freq=100000 \
    -o Engine.mix_precision.enable=False \

cp log_newir/workerlog.0 log_newir/workerlog_0.log
cp log_newir/workerlog.1 log_newir/workerlog_1.log
