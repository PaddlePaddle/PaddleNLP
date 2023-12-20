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

set -ex

export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH

export FLAGS_call_stack_level=2
export GLOG_logtostderr=true
export GLOG_v=0

export FLAGS_control_flow_use_new_executor=1
export FLAGS_new_executor_serial_run=1
export FLAGS_allocator_strategy=naive_best_fit
export FLAGS_fraction_of_gpu_memory_to_use=0.92

model_dir=${1:-"checkpoints/llama65b_ptq_smooth"}
src_len=${2:-1024}
dec_len=${3:-1024}
quant_type=${4:-"a8w8"}
# quant_type=${4:-"None"}

total_len=`expr ${src_len} + ${dec_len}`

python -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    predictor.py \
    --model_name_or_path ${model_dir} \
    --dtype float16 \
    --src_length ${total_len} \
    --max_length ${dec_len} \
    --output_file "infer.json" \
    --mode "dynamic" \
    --batch_size 2 \
    --inference_model \
    --quant_type ${quant_type}  \
    --block_attn \
    --use_cachekv_int8 static

