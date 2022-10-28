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

# GPU启动，参数`--gpus`指定训练所用的GPU卡号，可以是单卡，也可以多卡
unset CUDA_VISIBLE_DEVICES

python -m paddle.distributed.launch --gpus "2,3,4,5,6,7" run_summarization.py \
    --model_name_or_path=IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese \
    --train_file train.json \
    --eval_file test.json \
    --output_dir pegesus_out \
    --max_source_length 128 \
    --max_target_length 64 \
    --num_train_epochs 20 \
    --logging_steps 1 \
    --save_steps 10000 \
    --train_batch_size 128 \
    --eval_batch_size 128 \
    --learning_rate 5e-5 \
    --warmup_proportion 0.02 \
    --weight_decay=0.01 \
    --device=gpu \
