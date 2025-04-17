# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

set -eux

export BATCH_SIZE=8
export LR=2e-5
export EPOCH=12

unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" run_duie.py \
                            --device gpu \
                            --seed 42 \
                            --do_train \
                            --data_path ./data \
                            --max_seq_length 128 \
                            --batch_size $BATCH_SIZE \
                            --num_train_epochs $EPOCH \
                            --learning_rate $LR \
                            --warmup_ratio 0.06 \
                            --output_dir ./checkpoints
