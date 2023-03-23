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

model_item=gpt_sp_False
dp_degree=2
mp_degree=8
pp_degree=2
bs_item=16
fp_item=fp16
run_mode=DP2-MP8-PP2
device_num=N4C32
sequence_parallel=False

model=gpt
micro_bs=8

cd ./benchmarks
bash ./test_tipc/gpt/dygraph/sequence_parallel/benchmark_common/prepare.sh
# run
bash ./test_tipc/gpt/dygraph/sequence_parallel/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_bs} ${bs_item} ${run_mode} ${device_num} \
${sequence_parallel} 2>&1;
