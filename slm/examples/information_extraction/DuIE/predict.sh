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

export CUDA_VISIBLE_DEVICES=0 
export BATCH_SIZE=64
export CKPT=./checkpoints/model_90000.pdparams
export DATASET_FILE=./data/test1.json

python run_duie.py \
    --do_predict \
    --init_checkpoint $CKPT \
    --predict_data_file $DATASET_FILE \
    --max_seq_length 128 \
    --batch_size $BATCH_SIZE

