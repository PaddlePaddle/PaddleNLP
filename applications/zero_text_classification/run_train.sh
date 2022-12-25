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

#device=$1

#python -m paddle.distributed.launch --gpus "5,6" train.py \
CUDA_VISIBLE_DEVICES=5 python -u train.py \
    --dataset_dir data_test \
    --output_dir ckpt_test \
    --shuffle_choices True \
    --do_train \
    --do_eval \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --eval_steps 1 \
    --save_steps 1 \
    --logging_steps 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --max_seq_length 2500 \
    --load_best_model_at_end \
    --metric_for_best_model micro_f1 \
    --save_total_limit 1 \
    --warmup_ratio 0.1
    
    
    --disable_tqdm True \
    --train_file multi-task-v2/catslu_traindev_train.txt \
