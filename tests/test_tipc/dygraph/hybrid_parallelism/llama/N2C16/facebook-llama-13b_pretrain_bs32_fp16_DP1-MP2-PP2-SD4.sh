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

model_name_or_path="facebook/llama-13b"
per_device_train_batch_size="2"
use_flash_attention="1"
tensor_parallel_degree="2"
pipeline_parallel_degree="2"
virtual_pp_degree="1"
sequence_parallel="0"
sharding_degree="2"
num_train_epochs="1"
save_steps="200"
sharding="stage1"
recompute="1"
run_mode="DP1-MP2-PP2-SD4"
device_num="N2C16"
global_batch_size=32
model_item="facebook-llama-13b_pretrain"
max_step=80
gradient_accumulation_steps=16
pp_recompute_interval=1

cd ./tests
bash ./test_tipc/dygraph/hybrid_parallelism/llama/benchmark_common/prepare.sh

bash ./test_tipc/dygraph/hybrid_parallelism/llama/benchmark_common/run_benchmark.sh ${model_name_or_path} ${per_device_train_batch_size} ${use_flash_attention} ${tensor_parallel_degree} ${pipeline_parallel_degree} ${virtual_pp_degree} ${sequence_parallel} ${sharding_degree} ${num_train_epochs} ${save_steps} ${sharding} ${recompute} ${run_mode} ${device_num} ${global_batch_size} ${model_item} ${max_step} ${gradient_accumulation_steps} ${pp_recompute_interval}
