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

device=$1
test_path=$2
ckpt=$3
num_labels=$4

CUDA_VISIBLE_DEVICES=$device python predict.py \
    --test_path $test_path \
    --model_state $ckpt \
    --output_dir ckpt_test \
    --per_device_eval_batch_size 2 \
    --max_seq_length 1024 \
    --num_labels $num_labels
    
    
