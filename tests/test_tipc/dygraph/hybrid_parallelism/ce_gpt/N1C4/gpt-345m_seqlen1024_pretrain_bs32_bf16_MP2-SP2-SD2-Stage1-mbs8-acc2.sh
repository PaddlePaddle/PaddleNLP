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

model_item=gpt-345m_seqlen1024_pretrain
dp_degree=1
mp_degree=2
pp_degree=1
bs_item=32
fp_item=bf16
run_mode=MP2-SP2-SD2-Stage1-mbs8-acc2
device_num=N1C4
max_iter=100
sharding=stage1
sharding_degree=2

virtual_pp_degree=1
use_recompute=True
eval_freq=25
use_pipeline_parallel=False
sequence_parallel=True

model=gpt
micro_bs=8

bash ./test_tipc/dygraph/hybrid_parallelism/ce_gpt/benchmark_common/prepare.sh
# run
bash ./test_tipc/dygraph/hybrid_parallelism/ce_gpt/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_bs} ${bs_item} ${run_mode} ${device_num} \
${max_iter} ${sharding} ${sharding_degree} ${virtual_pp_degree} ${use_recompute} ${eval_freq} ${use_pipeline_parallel} ${sequence_parallel} 2>&1;