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

unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "2,3,4,5" run_pretrain.py \
    --input_dir ./data \
    --output_dir ./output \
    --learning_rate 1e-7 \
    --batch_size 2 \
    --adam_epsilon 1e-8 \
    --weight_decay 1e-2 \
    --warmup_steps 10000 \
    --max_steps 1000000 \
    --save_steps 10000 \
    --logging_steps 1 \
    --seed 1000 \
    --use_amp