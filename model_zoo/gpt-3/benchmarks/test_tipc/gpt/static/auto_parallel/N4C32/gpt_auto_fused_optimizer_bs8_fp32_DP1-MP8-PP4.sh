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

model_item=gpt_auto_fused_optimizer
fp_item=fp16
dp_degree=1
mp_degree=8
pp_degree=4
micro_batch_size=8
global_batch_size=8
run_mode=DP1-MP8-PP1
device_num=N1C8
max_iter=1000
use_recompute=False
verbose=3
logging_freq=1
use_passes=True

model=gpt

cd ./benchmarks
bash ./test_tipc/gpt/static/auto_parallel/benchmark_common/prepare.sh
# run
bash ./test_tipc/gpt/static/auto_parallel/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_batch_size} ${global_batch_size} ${run_mode} ${device_num} \
${max_iter} ${use_recompute} ${use_passes} ${verbose} ${logging_freq} 2>&1;