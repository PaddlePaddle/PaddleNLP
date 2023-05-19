# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT

nohup python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" train_vae.py \
    --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 \
    --ignore_keys decoder. \
    --vae_config_file config/vae.json \
    --freeze_encoder \
    --input_size 256 256 \
    --max_train_steps 100000000000 \
    --learning_rate 1e-4 \
    --batch_size 4 \
    --num_workers 8 \
    --logging_steps 100 \
    --save_steps 4000 \
    --image_logging_steps 2000 \
    --disc_start 50001 \
    --kl_weight 0.000001 \
    --disc_weight 0.5 \
    --resolution 512 1> paddle_vae.out 2>&1 & 