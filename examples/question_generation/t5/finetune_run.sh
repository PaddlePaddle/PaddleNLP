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

python -m paddle.distributed.launch --gpus 4,5,6,7 finetune.py \
    --model_name_or_path=t5-base \
    --dataset_name=squad \
    --output_dir=output \
    --max_source_length=1024 \
    --max_target_length=142 \
    --learning_rate=1e-4 \
    --num_train_epochs=6 \
    --logging_steps=100 \
    --save_steps=1000 \
    --seed=42 \
    --train_batch_size=8 \
    --eval_batch_size=64 \
    --warmup_proportion=0.1 \
    --device=gpu