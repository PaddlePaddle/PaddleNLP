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

export FLAGS_call_stack_level=2
export GLOG_logtostderr=true
export GLOG_v=0

export FLAGS_control_flow_use_new_executor=1
export FLAGS_new_executor_serial_run=1
export FLAGS_allocator_strategy=naive_best_fit
export FLAGS_fraction_of_gpu_memory_to_use=0.92
export CUDA_VISIBLE_DEVICES=0

# python predictor.py \
#     --model_name_or_path /root/paddlejob/workspace/env_run/lzy/paddlenlp_all_version/paddlenlp_internal/PaddleNLP/llm/guojihua \
#     --dtype float16 \
#     --src_length 1024 \
#     --max_length 1024 \
#     --output_file "infer.json" \
#     --mode "dynamic" \
#     --batch_size 1 \
#     --block_attn \
#     --inference_model

# linly-ai/chinese-llama-2-7b
# guojihua

python predictor.py \
    --model_name_or_path linly-ai/chinese-llama-2-7b \
    --dtype float16 \
    --src_length 1024 \
    --max_length 1024 \
    --output_file "infer.json" \
    --mode "dynamic" \
    --batch_size 2 \
    --inference_model \
    --block_attn 
    # --export_precache \
    # --prefix_path "/root/paddlejob/workspace/env_run/lzy/PaddleNLP/llm/ptuning-embedding/8-test/1/task_prompt_embeddings.npy"
    # --use_cachekv_int8 \
    # --quant_type weight_only_int8