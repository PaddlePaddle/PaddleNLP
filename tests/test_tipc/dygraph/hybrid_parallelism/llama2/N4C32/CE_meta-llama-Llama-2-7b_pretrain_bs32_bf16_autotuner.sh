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

# model_item包含CE_表示执行autotuner的脚本，无任务超时自动终止逻辑
param="model_item=CE_meta-llama-Llama-2-7b_pretrain "
# run_mode设置为autotuner，执行命令会匹配--auto_tuner_json，自动执行autotuner
param+="run_mode=autotuner "
param+="device_num=N4C32 "
param+="global_batch_size=32 "
param+="nnodes=4 "
param+="model_type=llama2_7b "

cd ./tests
bash ./test_tipc/dygraph/hybrid_parallelism/llama2/benchmark_common/prepare.sh

bash -c "${param} bash ./test_tipc/dygraph/hybrid_parallelism/llama2/benchmark_common/run_benchmark.sh"
