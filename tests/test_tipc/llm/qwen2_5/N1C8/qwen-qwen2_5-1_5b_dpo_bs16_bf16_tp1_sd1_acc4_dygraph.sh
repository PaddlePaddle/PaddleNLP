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


param="model_name_or_path=Qwen/Qwen2.5-1.5B "
param+="per_device_train_batch_size=1 "
param+="tensor_parallel_degree=1 "
param+="sharding_parallel_degree=1 "
param+="gradient_accumulation_steps=4 "
param+="run_stage=dpo "
param+="run_mode=tp1_sd1_acc4_dygraph "
param+="device_num=N1C8 "
param+="global_batch_size=16 "
param+="model_item=qwen-qwen2_5-1_5b_dpo "

cd ./tests
bash ./test_tipc/llm/qwen2_5/benchmark_common/prepare.sh

bash -c "${param} bash ./test_tipc/llm/qwen2_5/benchmark_common/run_benchmark.sh"
