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

model_name_or_path="__internal_testing__/fake-gpt-13b"
model_item="fake-gpt-13b_seqlen1024_pretrain"
tokenizer_name_or_path="__internal_testing__/fake-gpt-13b"
max_seq_length=1024
per_device_train_batch_size=4
per_device_eval_batch_size=16
tensor_parallel_degree=2
pipeline_parallel_degree=2
fuse_attention_qkv=1
use_flash_attention=1
fp16_opt_level="O2" 
max_steps=200
dataloader_num_workers=1
sharding="stage1"
sharding_parallel_degree=4
recompute=1
gradient_accumulation_steps=4
run_mode="DP1-MP2-PP2-SD4-stage1-mbs4-acc4-recompute"
device_num="N2C16"
global_batch_size=64

cd ./tests
bash ./test_tipc/dygraph/hybrid_parallelism/llm-gpt-3/benchmark_common/prepare.sh

per_device_eval_batch_size=${per_device_eval_batch_size} sharding_parallel_degree=${sharding_parallel_degree} run_mode=${run_mode} device_num=${device_num} global_batch_size=${global_batch_size} model_name_or_path=${model_name_or_path} model_item=${model_item} tokenizer_name_or_path=${tokenizer_name_or_path} max_seq_length=${max_seq_length} per_device_train_batch_size=${per_device_train_batch_size} tensor_parallel_degree=${tensor_parallel_degree} pipeline_parallel_degree=${pipeline_parallel_degree} fuse_attention_qkv=${fuse_attention_qkv} use_flash_attention=${use_flash_attention} fp16_opt_level=${fp16_opt_level} max_steps=${max_steps} dataloader_num_workers=${dataloader_num_workers} sharding=${sharding} recompute=${recompute} gradient_accumulation_steps=${gradient_accumulation_steps} bash ./test_tipc/dygraph/hybrid_parallelism/llm-gpt-3/benchmark_common/run_benchmark.sh

fake-gpt-13b_seqlen1024_pretrain_bs64_fp16_DP1-MP2-PP2-SD4-stage1-mbs4-acc4-recompute.sh