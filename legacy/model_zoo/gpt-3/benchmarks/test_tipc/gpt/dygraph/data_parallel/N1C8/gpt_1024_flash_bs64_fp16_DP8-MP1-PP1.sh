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

model_item=gpt_1024_flash
dp_degree=8
mp_degree=1
pp_degree=1
bs_item=64
fp_item=fp16
run_mode=DP8-MP1-PP1
device_num=N1C8
yaml_path=./ppfleetx/configs/nlp/gpt/pretrain_gpt_345M_single_card.yaml

model=gpt
micro_bs=8

cd ./benchmarks
bash ./test_tipc/gpt/dygraph/data_parallel/benchmark_common/prepare.sh
# run
bash ./test_tipc/gpt/dygraph/data_parallel/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_bs} ${bs_item} ${run_mode} ${device_num} \
${yaml_path} 2>&1;
