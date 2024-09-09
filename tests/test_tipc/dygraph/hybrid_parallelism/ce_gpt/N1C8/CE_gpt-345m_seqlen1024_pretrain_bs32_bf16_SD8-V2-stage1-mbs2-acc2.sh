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
dp_degree=1
mp_degree=1
pp_degree=1
sharding_degree=8
sharding=stage1
bs_item=32
fp_item=bf16
run_mode=SD8-V2-stage1-mbs2-acc2
device_num=N1C8
max_iter=50000
virtual_pp_degree=1
use_recompute=True
eval_freq=25
use_pipeline_parallel=False
sequence_parallel=False
acc=2
seed=1234
sharding_v2=True

model=gpt
micro_bs=2

bash ./test_tipc/dygraph/hybrid_parallelism/ce_gpt/benchmark_common/prepare.sh
# run
bash ./test_tipc/dygraph/hybrid_parallelism/ce_gpt/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_bs} ${bs_item} ${run_mode} ${device_num} \
${max_iter} ${sharding} ${sharding_degree} ${virtual_pp_degree} ${use_recompute} ${eval_freq} ${use_pipeline_parallel} ${sequence_parallel} ${acc} ${seed} ${sharding_v2} 2>&1;