# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

param="model_item=meta-llama-Llama-2-7b_pretrain_dy2st "
param+="run_mode=Sharding_Stage1 "
param+="device_num=N1C8 "
param+="global_batch_size=8 "
param+="nnodes=1 "
param+="model_type=llama2_7b "

cd ./tests
bash ./test_tipc/static/auto_parallel/llama2/benchmark_common/prepare.sh

bash -c "${param} bash ./test_tipc/static/auto_parallel/llama2/benchmark_common/run_benchmark.sh"
