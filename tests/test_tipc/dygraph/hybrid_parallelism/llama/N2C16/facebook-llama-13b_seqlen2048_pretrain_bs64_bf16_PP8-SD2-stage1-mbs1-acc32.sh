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

param="model_name_or_path=facebook/llama-13b "
param+="per_device_train_batch_size=1 "
param+="data_parallel_degree=1 "
param+="tensor_parallel_degree=1 "
param+="pipeline_parallel_degree=8 "
param+="virtual_pp_degree=1 "
param+="sequence_parallel=0 "
param+="sharding_parallel_degree=2 "
param+="sharding=stage1 "
param+="recompute=0 "
param+="run_mode=PP8-SD2-stage1-mbs1-acc32 "
param+="device_num=N2C16 "
param+="global_batch_size=64 "
param+="model_item=facebook-llama-13b_seqlen2048_pretrain "
param+="max_steps=150 "
param+="gradient_accumulation_steps=32 "
param+="pp_recompute_interval=1 "
param+="tensor_parallel_config=enable_mp_async_allreduce,enable_mp_skip_c_identity,enable_mp_fused_linear_param_grad_add "
param+="recompute_use_reentrant=true "

cd ./tests
bash ./test_tipc/dygraph/hybrid_parallelism/llama/benchmark_common/prepare.sh

bash -c "${param} bash ./test_tipc/dygraph/hybrid_parallelism/llama/benchmark_common/run_benchmark.sh"
