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
intokens_max_length=4096
learning_rate="3e-05"
recompute="1"
tensor_parallel_degree="8"
lora="0"
prefix_tuning="0"
model_item="facebook-llama-7b_sft"
run_mode="MP8"
device_num="N1C8"

cd ./tests
bash ./test_tipc/llm/benchmark/benchmark_common/prepare.sh
bash ./test_tipc/llm/benchmark/benchmark_common/run_benchmark.sh ${model_name_or_path} ${dataset_name_or_path} ${intokens_max_length} ${learning_rate} ${recompute} ${tensor_parallel_degree} ${lora} ${prefix_tuning} ${model_item} ${run_mode} ${device_num} 