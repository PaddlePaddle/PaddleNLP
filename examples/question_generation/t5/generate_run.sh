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

python generate.py \
    --model_name_or_path=mrm8488/t5-base-finetuned-question-generation-ap \
    --dataset_name=squad \
    --output_path=generate.txt \
    --max_source_length=1024 \
    --max_target_length=142 \
    --decode_strategy=greedy_search \
    --top_k=2 \
    --top_p=1.0 \
    --num_beams=1 \
    --length_penalty=0.0 \
    --batch_size=64 \
    --seed=42 \
    --logging_steps=20 \
    --device=gpu