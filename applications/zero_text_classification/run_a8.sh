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
python -m paddle.distributed.launch --gpus "2,5,6" train.py \
    --dataset_dir data_v4 \
    --output_dir ckpt_v4_12271738 \
    --shuffle_choices True \
    --do_train \
    --do_eval \
    --num_train_epochs 10 \
    --learning_rate 2e-5 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --logging_steps 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --max_seq_length 1024 \
    --load_best_model_at_end \
    --metric_for_best_model macro_f1 \
    --save_total_limit 1 \
    --disable_tqdm True \
    --warmup_ratio 0.1
    
    
    #--train_file multi-task-v2/catslu_traindev_train.txt \
