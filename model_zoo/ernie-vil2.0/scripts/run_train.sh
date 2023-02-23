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

DATAPATH=./data

# data options
train_data=${DATAPATH}/datasets/Flickr30k-CN/lmdb/train
val_data=${DATAPATH}/datasets/Flickr30k-CN/lmdb/valid

# --test_only \
log_dir=train_log
python -u -m paddle.distributed.launch --gpus "0,1" \
                --log_dir ${log_dir}  \
                run_finetune.py --output_dir output_pd \
                --do_train \
                --train_data=${train_data} \
                --val_data=${val_data} \
                --learning_rate 5e-5 \
                --warmup_steps 100 \
                --logging_steps 50 \
                --per_device_train_batch_size 128 \
                --dataloader_num_workers 8 \
                --save_steps 50 \
                --num_train_epochs 32 \
                --weight_decay 0.001 \
                --save_total_limit 50 \
                --seed 1 \
                --label_names index \
                --data_root ./data \
                --lr_scheduler_type cosine \
                --recompute