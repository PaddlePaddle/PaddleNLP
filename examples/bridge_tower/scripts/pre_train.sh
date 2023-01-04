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

export PYTHONPATH=/root/paddlejob/workspace/env_run/wugaosheng/PaddleNLP:$PYTHONPATH
# 屏蔽平台预设的环境变量，因为框架采用兼容升级，检测到这些配置会使用原方式启动
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT

# python -u -m paddle.distributed.launch --ips 10.78.122.11,10.78.115.13 \
#                 --gpus "0,1,2,3,4,5,6,7" \
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" \
                python train.py --output_dir pretrain_output \
                --do_train \
                --do_eval \
                --evaluation_strategy epoch \
                --learning_rate 1.0 \
                --eval_steps 1000 \
                --warmup_ratio 0.1 \
                --save_steps 1000 \
                --dataloader_num_workers 8 \
                --logging_steps 50 \
                --per_device_train_batch_size 16 \
                --per_device_eval_batch_size 128 \
                --batch_size 128 \
                --num_train_epochs 10 \
                --weight_decay 0.01 \
                --save_total_limit 50 \
                --adam_epsilon 1e-8 \
                --adam_beta1 0.9 \
                --adam_beta2 0.98 \
                --lr_scheduler_type polynomial \
                --lr_end 0 \
                --seed 1 \
                --max_grad_norm -1 \
                --max_steps 100000 \
                --label_names text_labels \
                --data_root /root/paddlejob/workspace/env_run/afs/laion400m_new/wugaosheng/dataset/pre-train \
                --power 1 \
                --config_name configs/pretrain_config.json

# python train.py --output_dir pretrain_output \
#                 --do_train \
#                 --learning_rate 1e-05 \
#                 --eval_steps 100 \
#                 --warmup_ratio 0.1 \
#                 --logging_steps 50 \
#                 --per_device_train_batch_size 16 \
#                 --per_device_eval_batch_size 128 \
#                 --num_train_epochs 10 \
#                 --weight_decay 0.01 \
#                 --adam_epsilon 1e-8 \
#                 --adam_beta1 0.9 \
#                 --adam_beta2 0.98 \
#                 --lr_scheduler_type polynomial \
#                 --lr_end 0 \
#                 --max_steps 100000 \
#                 --power 1 \
#                 --config_name configs/pretrain_config.json



