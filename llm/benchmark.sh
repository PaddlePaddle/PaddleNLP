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

export FLAGS_control_flow_use_new_executor=1
export FLAGS_new_executor_serial_run=1
export FLAGS_allocator_strategy=naive_best_fit
export FLAGS_fraction_of_gpu_memory_to_use=0.92

export FLAGS_use_autotune=1
export FLAGS_cublaslt_exhaustive_search_times=10

export CUDA_VISIBLE_DEVICES=1

# dygraph

for bsz in 1 2;do
python predictor.py \
    --model_name_or_path "/root/paddlejob/workspace/env_run/wufeisheng/paddlenlp_ckpt/checkpoints/llama_ptq_ckpts" \
    --dtype float16 \
    --src_length 300 \
    --max_length 100 \
    --mode "dynamic" \
    --quant_type "A8W8" \
    --inference_model \
    --shift_smooth 1 \
    --batch_size ${bsz} \
    --benchmark
done

# inference statc=ic graph
# for bsz in 1 2 4 8 16;do
# python predictor.py \
#     --model_name_or_path ./inference_ptq \
#     --dtype float16 \
#     --src_length 300 \
#     --max_length 100 \
#     --output_file "infer.json" \
#     --mode "static" \
#     --batch_size ${bsz} \
#     --benchmark \
#     --inference_model 
# done