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

export CUDA_VISIBLE_DEVICES=3

for model_name in facebook/opt-125m facebook/opt-350m;
    do   
        for top_k in 1 4 8 16;
            do
                python opt_perf.py \
                    --model_name_or_path=$model_name \
                    --decode_strategy=sampling \
                    --top_k=$top_k \
                    --top_p=0.4 \
                    --max_length=32 
                sleep 10s
                python opt_perf.py \
                    --model_name_or_path=$model_name \
                    --decode_strategy=sampling \
                    --top_k=$top_k \
                    --top_p=0.4 \
                    --max_length=32 \
                    --use_fp16_decoding
                sleep 10s
            done
        python opt_perf.py \
            --model_name_or_path=$model_name \
            --decode_strategy=sampling \
            --top_k=0 \
            --top_p=0.4 \
            --max_length=32 
        sleep 10s
        python opt_perf.py \
            --model_name_or_path=$model_name \
            --decode_strategy=sampling \
            --top_k=0 \
            --top_p=0.4 \
            --max_length=32 \
            --use_fp16_decoding
        sleep 10s
    done
