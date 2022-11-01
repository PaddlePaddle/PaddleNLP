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

model_item=bert_large_seqlen512
model=bert
bs_item=4
fp_item=fp16
run_mode=DP
device_num=N1C1
max_epochs=1000
num_workers=0

# get data
bash ./test_tipc/static/dp/${model}/benchmark_common/prepare.sh
# run
bash ./test_tipc/static/dp/${model}/benchmark_common/run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num} ${max_epochs} ${num_workers} 2>&1;
