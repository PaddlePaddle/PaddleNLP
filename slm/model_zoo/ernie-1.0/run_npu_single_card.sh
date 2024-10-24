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

set -x

export FLAGS_selected_npus=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export FLAGS_allocator_strategy=naive_best_fit

rm -rf core.*

task_name="ernie-1.0-npu"
rm -rf output/$task_name/log

python -u run_pretrain.py \
    --model_type "ernie" \
    --model_name_or_path "ernie-3.0-base-zh" \
    --tokenizer_name_or_path "ernie-3.0-base-zh" \
    --input_dir "./data" \
    --data_impl "mmap" \
    --output_dir "output/$task_name" \
    --split 949,50,1 \
    --max_seq_len 512 \
    --micro_batch_size 52 \
    --use_amp true \
    --fp16_opt_level "O1" \
    --use_recompute false \
    --max_lr 0.0001 \
    --min_lr 0.00001 \
    --max_steps 1000000 \
    --save_steps 50000 \
    --checkpoint_steps 5000 \
    --decay_steps 990000 \
    --weight_decay 0.01 \
    --warmup_rate 0.01 \
    --grad_clip 1.0 \
    --logging_freq 20\
    --num_workers 8 \
    --eval_freq 1000 \
    --device "npu"\
