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

param="model_item=CE_buffer_autotuner_llama7b "
param+="run_mode=pretrain "
param+="device_num=N1C8 "
param+="global_batch_size=8 "
param+="autoconfig_json_file=autoconfig/llama7b_pretrain_buffer.json "
param+="modle_json_file=autoconfig/llama7b_pretrain_params.json "

cd ./tests
bash ./test_tipc/auto_tuner/llama_pretrain/benchmark_common/prepare.sh

bash -c "${param} bash ./test_tipc/auto_tuner/llama_pretrain/benchmark_common/run_benchmark.sh"
