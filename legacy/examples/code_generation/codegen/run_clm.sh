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

python -m paddle.distributed.launch --gpus 0,1 run_clm.py \
            --model_name_or_path Salesforce/codegen-350M-mono \
            --block_size 1024 \
            --output_dir output \
            --train_file train.json \
            --validation_file test.json \
            --num_train_epochs 5 \
            --logging_steps 10 \
            --save_steps 1000 \
            --per_device_train_batch_size 2 \
            --per_device_eval_batch_size 2 \
            --learning_rate 1e-4 \
            --warmup_ratio 0.1 \
            --do_train \
            --do_eval \
            --device gpu
