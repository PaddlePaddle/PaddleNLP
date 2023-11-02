# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH

# python -m paddle.distributed.launch \
#     --gpus "0" \
#      export_model.py \
#     --model_name_or_path /root/paddlejob/workspace/env_run/wufeisheng/paddlenlp_ckpt/checkpoints/llama_sft_ckpts/checkpoint-10 \
#     --output_path ./llama13b-inference_model_fp16_mp1 \
#     --dtype float16 \
#     --inference_model \
#     --block_attn

# python -m paddle.distributed.launch \
#     --gpus "0" \
#      export_model.py \
#     --model_name_or_path /root/paddlejob/workspace/env_run/wufeisheng/paddlenlp_ckpt/checkpoints/llama_sft_ckpts/checkpoint-10 \
#     --output_path ./llama13b-inference_model_fp16_mp1_cachekvint8 \
#     --dtype float16 \
#     --inference_model \
#     --block_size 16 \
#     --block_attn \
#     --use_cachekv_int8


python \
     export_model.py \
    --model_name_or_path meta-llama/Llama-2-13b \
    --output_path ./llama-13b_inference_model_wint8_mp1_52 \
    --dtype float16 \
    --inference_model \
    --block_size 64 \
    --src_length 1024 \
    --block_attn \
    --quant_type weight_only_int8 
    
    
    
#     --use_cachekv_int8


    # --export_precache \
    # --prefix_path "/root/paddlejob/workspace/env_run/lzy/PaddleNLP/llm/ptuning-embedding/8-test/1/task_prompt_embeddings.npy"