# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

model_item=CE_gpt-345m_seqlen1024_pretrain
dp_degree=2
mp_degree=2
pp_degree=4
bs_item=32
fp_item=bf16
run_mode=MP2-SP2-PP4-VPP2-DP2-mbs2-acc8
device_num=N2C16
max_iter=50000
sharding=False
sharding_degree=1

virtual_pp_degree=2
use_recompute=True
eval_freq=25
use_pipeline_parallel=True
sequence_parallel=True

model=gpt
micro_bs=2
acc=8
seed=1234

bash ./test_tipc/dygraph/hybrid_parallelism/ce_gpt/benchmark_common/prepare.sh
# run
bash ./test_tipc/dygraph/hybrid_parallelism/ce_gpt/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_bs} ${bs_item} ${run_mode} ${device_num} \
${max_iter} ${sharding} ${sharding_degree} ${virtual_pp_degree} ${use_recompute} ${eval_freq} ${use_pipeline_parallel} ${sequence_parallel} ${acc} ${seed} 2>&1;