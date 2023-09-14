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
dataset_name_or_path="llm_benchmark_en"
base_batch_size=1
learning_rate="3e-02"
recompute="1"
tensor_parallel_degree="1"
lora="0"
prefix_tuning="1"
model_item="facebook-llama-13b_pt"
run_mode="DP1-recompute"
device_num="N1C1"
num_train_epochs=2
export CUDA_VISIBLE_DEVICES=0
cd ./tests
bash ./test_tipc/dygraph/ft/benchmark_common/prepare.sh
bash ./test_tipc/dygraph/ft/benchmark_common/run_benchmark.sh ${model_name_or_path} ${dataset_name_or_path} ${base_batch_size}  ${learning_rate} ${recompute} ${tensor_parallel_degree} ${lora} ${prefix_tuning} ${model_item} ${run_mode} ${device_num} ${num_train_epochs}