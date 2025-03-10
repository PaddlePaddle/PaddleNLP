# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

export PYTHONPATH=../../../:$PYTHONPATH # PaddleNLP dir
export PYTHONPATH=../../:$PYTHONPATH # PaddleNLP/llm dir
export USE_FAST_TOKENIZER=true
export NCCL_ALGO=Tree

# 设置为相对路径
# MODEL_PATH="Qwen/Qwen2-7B"
MODEL_PATH="Qwen/Qwen2.5-Math-7B"
# MODEL_PATH="Qwen/Qwen2.5-Math-1.5B-Instruct"
# MODEL_PATH="Qwen/Qwen2.5-7B-Instruct-1M"
# MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

mkdir -p results/${MODEL_PATH} && touch results/${MODEL_PATH}/output.json

python -u -m paddle.distributed.launch \
    --devices "0,1,2,3" \
    ../../predict/predictor.py \
    --model_name_or_path ${MODEL_PATH} \
    --inference_model true \
    --dtype bfloat16 \
    --batch_size 64 \
    --use_flash_attention true \
    --max_length 2048 \
    --total_max_length 4096 \
    --decode_strategy greedy_search \
    --temperature 0.95 \
    --data_file ./data/gsm8k/dev.json \
    --output_file results/${MODEL_PATH}/output.json
