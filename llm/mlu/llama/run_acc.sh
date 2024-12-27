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
set -ex

sed -i 's/"sequence_parallel": true/"sequence_parallel": false/' ./checkpoints/llama_pretrain_ckpts/checkpoint-5000/config.json

python ../../../legacy/examples/benchmark/wiki_lambada/eval.py --model_name_or_path ./checkpoints/llama_pretrain_ckpts/checkpoint-5000/ \
                                               --batch_size 4 \
                                               --eval_path ./wiki_lambada/lambada_test.jsonl \
                                               --tensor_parallel_degree 1 --cloze_eval 2>&1 | tee log_lambada_eval