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

python3.8 eval.py \
    --model_name_or_path 'qwen/qwen-7b-chat' \
    --cot False \
    --few_shot True \
    --with_prompt False \
    --constrained_decoding True \
    --temperature 0.2 \
    --n_times 1 \
    --ntrain 5 \
    --do_ptq \
    --do_smooth \
    --do_awq \
    --do_int4 \
    --do_autoclip \
    --do_save_csv False \
    --do_test False \
    --weight_quant_method "groupwise"\
    --dtype bfloat16 \
    --output_dir 'qwen/qwen-7b-chat-4bit-new-group' \


# "abs_max_channel_wise"