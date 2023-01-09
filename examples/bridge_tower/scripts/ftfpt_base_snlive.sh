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

# 屏蔽平台预设的环境变量，因为框架采用兼容升级，检测到这些配置会使用原方式启动
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT

# export NCCL_IB_DISABLE=0
export PYTHONPATH=/root/paddlejob/workspace/env_run/wugaosheng/PaddleNLP:$PYTHONPATH
# 启动方式
# export PADDLE_MASTER=10.78.122.11:2379 # etcd://10.11.60.193:2379
# export PADDLE_NNODES=4
# export CUDA_VISIBLE_DEVICES=0,1

# --logging_steps 50 \
# snli_ve_config vqav2_config
# python train.py --output_dir output \
#                 --learning_rate 1e-05 \
#                 --warmup_ratio 0.06 \
#                 --logging_steps 50 \
#                 --per_device_eval_batch_size 64 \
#                 --per_device_train_batch_size 8 \
#                 --num_train_epochs 10 \
#                 --weight_decay 0.05 \
#                 --adam_epsilon 1e-8 \
#                 --adam_beta1 0.9 \
#                 --adam_beta2 0.98 \
#                 --lr_scheduler_type polynomial \
#                 --lr_end 0 \
#                 --power 1 \
#                 --config_name configs/snli_ve_config.json

# snli_ve_config
# python train.py --output_dir output \
#                 --do_eval \
#                 --do_train \
#                 --learning_rate 3e-06 \
#                 --eval_steps 100 \
#                 --warmup_ratio 0.06 \
#                 --logging_steps 50 \
#                 --per_device_train_batch_size 8 \
#                 --per_device_eval_batch_size 64 \
#                 --num_train_epochs 5 \
#                 --weight_decay 0.01 \
#                 --adam_epsilon 1e-8 \
#                 --adam_beta1 0.9 \
#                 --adam_beta2 0.98 \
#                 --lr_scheduler_type polynomial \
#                 --lr_end 0 \
#                 --power 1 \
#                 --config_name configs/snli_ve_config.json

# export FLAGS_call_stack_level=2
# 多卡测试
# --disable_tqdm 1 \
# "0,1,2,3,4,5,6,7"
# --label_names  \
# python -u -m paddle.distributed.launch --ips 10.78.115.13 \
                # --gpus "0,1,2,3,4,5,6,7" \
# --disable_tqdm 1 \

# --dataloader_num_workers 8 \
# pretrained/BridgeTower_pt_base.zip.ckpt.pdparams
# --disable_tqdm 1 \
# learning_rate 1.0 paddle的分层学习率跟torch不一样，在使用分层学习率的时候，base的学习率需要设置成1.0
# rm -rf runs/
log_dir=train_log
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" \
                --log_dir ${log_dir}  \
                train.py --output_dir output_pd \
                --do_eval \
                --do_train \
                --learning_rate 1.0 \
                --warmup_ratio 0.06 \
                --logging_steps 50 \
                --eval_steps 100 \
                --evaluation_strategy epoch \
                --num_nodes 1 \
                --num_gpus 8 \
                --per_device_train_batch_size 8 \
                --per_device_eval_batch_size 64 \
                --batch_size 64 \
                --dataloader_num_workers 8 \
                --save_steps 8000 \
                --num_train_epochs 5 \
                --weight_decay 0.01 \
                --save_total_limit 50 \
                --adam_epsilon 1e-8 \
                --adam_beta1 0.9 \
                --adam_beta2 0.98 \
                --seed 1 \
                --data_root /root/paddlejob/workspace/env_run/afs/laion400m_new/wugaosheng/dataset/fine-tune \
                --lr_scheduler_type polynomial \
                --checkpoint_path ./checkpoints/checkpoint-40000/model_state.pdparams \
                --lr_end 0 \
                --power 1 \
                --max_grad_norm -1 \
                --config_name configs/snli_ve_config.json
                