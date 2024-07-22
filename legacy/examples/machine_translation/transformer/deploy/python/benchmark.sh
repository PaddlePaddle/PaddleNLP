#!/bin/bash

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

model_dir=${1}
model=${2}
mkdir -p output_pipeline
log_path="output_pipeline"

for batch_size in "1" "2" "4"; do
    python inference.py \
        --config="../../configs/transformer.${model}.yaml" \
        --device cpu \
        --model_dir=${model_dir} \
        --batch_size=${batch_size} \
        --profile > ${log_path}/transformer_${model}_cpu_nomkl_bs${batch_size}_inference.log 2>&1

    for threads in "1" "6"; do
        python inference.py \
            --config="../../configs/transformer.${model}.yaml" \
            --model_dir=${model_dir} \
            --device cpu \
            --use_mkl True \
            --threads=${threads} \
            --batch_size=${batch_size} \
            --profile > ${log_path}/transformer_${model}_cpu_mkl_threads${threads}_bs${batch_size}_inference.log 2>&1 
    done

    python inference.py \
        --config="../../configs/transformer.${model}.yaml" \
        --model_dir=${model_dir} \
        --device gpu \
        --batch_size=${batch_size} \
        --profile > tee ${log_path}/transformer_${model}_gpu_bs${batch_size}_inference.log 2>&1 
done
