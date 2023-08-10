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

export PYTHONPATH=$PYTHONPATH:../../..

export FLAGS_cache_inference_while_scope=1
export FLAGS_call_stack_level=2
export GLOG_logtostderr=1
export GLOG_v=0
export FLAGS_control_flow_use_new_executor=1
export FLAGS_new_executor_serial_run=1
export FLAGS_allocator_strategy=naive_best_fit
export FLAGS_fraction_of_gpu_memory_to_use=0.95
export FLAGS_use_cutlass_fmha=1

python -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" \
           infer_generation.py \
           --model_dir=${1} \
           --model_prefix=llama 2>&1