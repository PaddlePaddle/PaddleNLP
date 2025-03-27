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

nohup python distill_data.py \
    --input_file "./data/gsm8k_zh/GSM8K_zh.jsonl" \
    --output_dir "./data/GSM8K_distilled_en" \
    --prompt_key "question" \
    --response_key "deepseek_r1_response" \
    --reasoning_key "deepseek_r1_reasoning" \
    --prompt_suffix "\nPlease reason step by step, and put your final answer within \\boxed{}." \
    --base_urls "http://192.168.0.1:9965/v1,http://192.168.0.2:9965/v1" \
    --model deepseek-r1:671b \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_tokens 32768 \
    --concurrency 16 > ./meta-math_gsm8k_en_distill.log 2>&1 &

nohup python distill_data.py \
    --input_file "./data/gsm8k_zh/GSM8K_zh.jsonl" \
    --output_dir "./data/GSM8K_distilled_zh" \
    --prompt_key "question_zh" \
    --response_key "deepseek_r1_response_zh" \
    --reasoning_key "deepseek_r1_reasoning_zh" \
    --prompt_suffix "\n请一步一步地推理，并将你的最终答案放在\boxed{}中。" \
    --base_urls "http://192.168.0.1:9965/v1,http://192.168.0.2:9965/v1" \
    --model deepseek-r1:671b \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_tokens 32768 \
    --concurrency 16 > ./meta-math_gsm8k_zh_distill.log 2>&1 &
