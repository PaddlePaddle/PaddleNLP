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

export CUDA_VISIBLE_DEVICES=0

python  train.py \
        --train_path "../data/ext_data/train.txt" \
        --dev_path "../data/ext_data/dev.txt" \
        --label_path "../data/ext_data/label.dict" \
        --num_epochs 10 \
        --batch_size 16 \
        --max_seq_len 256 \
        --learning_rate 5e-5 \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
        --warmup_proportion 0.1 \
        --log_steps 50 \
        --eval_steps 250 \
        --seed 1000 \
        --device "gpu" \
        --checkpoints "../checkpoints/ext_checkpoints/"
