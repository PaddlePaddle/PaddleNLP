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

# MODEL_PATH="Qwen/Qwen2.5-Math-7B"
# DTYPE="float32"
# DECODE_STRATEGY="greedy_search"
# TEMPERATURE=0.95
# TOP_P=0.6
# SRC_LENGTH=1024
# MAX_LENGTH=3072
# TOTAL_MAX_LENGTH=4096

# MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# DTYPE="float32"
# DECODE_STRATEGY="greedy_search"
# TEMPERATURE=0.95
# TOP_P=0.6
# SRC_LENGTH=1024
# MAX_LENGTH=15360
# TOTAL_MAX_LENGTH=16384

MODEL_PATH=checkpoints/Qwen/Qwen2.5-Math-7B/gsm8k_distilled
DTYPE="bfloat16"
DECODE_STRATEGY="greedy_search"
TEMPERATURE=0.95
TOP_P=0.6
SRC_LENGTH=1024
MAX_LENGTH=3072
TOTAL_MAX_LENGTH=4096

INPUT_FILE="./data/gsm8k/dev.json"
# INPUT_FILE="./data/aime2024/dev.json"
# INPUT_FILE="./data/math500/dev.json"

OUTPUT_DIR="results-gsm8k/${MODEL_PATH}"
# OUTPUT_DIR="results-aime2024/${MODEL_PATH}"
# OUTPUT_DIR="results-math500/${MODEL_PATH}"

mkdir -p ${OUTPUT_DIR} && touch ${OUTPUT_DIR}/output.json

python -u -m paddle.distributed.launch \
    --devices "0,1,2,3" \
    ../../predict/predictor.py \
    --model_name_or_path ${MODEL_PATH} \
    --inference_model true \
    --append_attn true \
    --dtype ${DTYPE} \
    --batch_size 4 \
    --use_flash_attention true \
    --src_length ${SRC_LENGTH} \
    --max_length ${MAX_LENGTH} \
    --total_max_length ${TOTAL_MAX_LENGTH} \
    --decode_strategy ${DECODE_STRATEGY} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --data_file ${INPUT_FILE} \
    --output_file ${OUTPUT_DIR}/output.json
