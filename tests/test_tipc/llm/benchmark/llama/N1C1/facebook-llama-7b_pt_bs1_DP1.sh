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

model_name_or_path="facebook/llama-7b"
dataset_name_or_path="llm_benchmark_en"
intokens_max_length=3072
learning_rates="3e-02"
recompute="1"
tensor_parallel_degree="1"
lora="0"
prefix_tuning="1"
model_item="facebook-llama-7b_pt"
run_mode="DP1"
device_num="N1C1"

cd ./tests
bash ./test_tipc/llm/benchmark/benchmark_common/prepare.sh
bash ./test_tipc/llm/benchmark/benchmark_common/run_benchmark.sh ${model_name_or_path} ${dataset_name_or_path} ${intokens_max_length} ${learning_rates} ${recompute} ${tensor_parallel_degree} ${lora} ${prefix_tuning} ${model_item} ${run_mode} ${device_num} 