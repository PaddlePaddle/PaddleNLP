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

param="model_item=baichuan-inc-Baichun2-13b_pretrain "
param+="run_mode=DP1_MP4_PP1_VPP1_Sharding8_Stage1 "
param+="device_num=N4C32 "
param+="global_batch_size=32 "
param+="nnodes=4 "
param+="model_type=baichun2_13b "

cd ./tests
bash ./test_tipc/dygraph/hybrid_parallelism/baichun2/benchmark_common/prepare.sh

bash -c "${param} bash ./test_tipc/dygraph/hybrid_parallelism/baichun2/benchmark_common/run_benchmark.sh"
