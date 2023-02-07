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
export CUDA_VISIBLE_DEVICES=0

python -u run_pretrain.py \
    --model_type "gpt"\
    --model_name_or_path gpt2-en \
    --input_dir ./data \
    --output_dir ./output_dir/pretrain \
    --weight_decay 0.01 \
    --max_steps 500000 \
    --save_steps 100000 \
    --warmup_steps 320000 \
    --warmup_ratio 0.01 \
    --per_device_train_batch_size 4 \
    --device gpu \
    --eval_steps 500 \
    --do_train true \
    --do_predict true
