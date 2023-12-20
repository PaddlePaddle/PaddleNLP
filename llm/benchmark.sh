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
export FLAGS_cache_inference_while_scope=1
export FLAGS_dynamic_static_unified_comm=0
# export FLAGS_benchmark=1

model_dir=${1:-"checkpoints/llama65b_ptq_smooth_mp8"}
src_len=${2:-1100}
dec_len=${3:-330}

total_len=`expr ${src_len} + ${dec_len}`


python -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    predictor.py \
    --model_name_or_path ./inference_model/${model_dir} \
    --dtype float16 \
    --src_length ${total_len} \
    --max_length ${dec_len} \
    --output_file "infer.json" \
    --mode "static" \
    --batch_size 128 \
    --benchmark \
    --block_attn \
    --block_size 64 \
    --inference_model \
    --use_cachekv_int8 static