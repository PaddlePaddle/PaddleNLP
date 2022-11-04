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

# Cuda device ids used
cuda_device=""
# Folder used to save checkpoints
save_dir=
# Data set path for training
train_path=
# Data set path for development
dev_path=

###############################################################################################################################################################################################################################################################################################################################################
python -u -m paddle.distributed.launch --gpus $cuda_device finetune.py \
            --train_path $train_path \
            --dev_path $dev_path \
            --save_dir $save_dir \
            --learning_rate 1e-5 \
            --batch_size 32 \
            --max_seq_len 512 \
            --num_epochs 30 \
            --model uie-base \
            --seed 1000 \
            --logging_steps 100 \
            --valid_steps 5000 \
            --device gpu