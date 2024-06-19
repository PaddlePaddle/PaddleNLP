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

set -x
unset CUDA_VISIBLE_DEVICES

rm -rf core.*

# dp8 for 8 worker of data parallel
# gb512 for the global batch size is 512 = 64 * 8
# s1m for max steps is 1 million
task_name="ernie-1.0-dp8-gb512"
rm -rf output/$task_name/log

python -u  -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir "output/$task_name/log" \
    run_pretrain.py \
    --model_type "ernie" \
    --model_name_or_path "ernie-1.0-base-zh" \
    --tokenizer_name_or_path "ernie-1.0-base-zh" \
    --input_dir "./data" \
    --data_impl "mmap" \
    --output_dir "output/$task_name" \
    --split 949,50,1 \
    --max_seq_len 512 \
    --micro_batch_size 64 \
    --use_amp true \
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
    --num_workers 2 \
    --eval_freq 1000 \
    --device "gpu"\

