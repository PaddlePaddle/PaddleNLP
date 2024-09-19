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


param="model_name_or_path=qwen/qwen-7b "
param+="per_device_train_batch_size=1 "
param+="data_parallel_degree=1 "
param+="tensor_parallel_degree=1 "
param+="pipeline_parallel_degree=1 "
param+="virtual_pp_degree=1 "
param+="sequence_parallel=0 "
param+="sharding_parallel_degree=32 "
param+="sharding=stage1 "
param+="recompute=0 "
param+="recompute_granularity=full_attn "
param+="run_mode=MP1-PP1-sharding32-mbs1-acc1 "
param+="device_num=N4C32 "
param+="global_batch_size=32 "
param+="model_item=qwen-qwen-7b_seqlen4096_pretrain "
param+="max_steps=200 "
param+="logging_steps=5 "
param+="gradient_accumulation_steps=1 "
param+="pp_recompute_interval=1 "
param+="tensor_parallel_config=enable_mp_async_allreduce,enable_mp_skip_c_identity,enable_mp_fused_linear_param_grad_add "
#多机新添加的参数
param+="pipeline_parallel_config=enable_sharding_comm_overlap,enable_release_grads "
param+="max_seq_length=4096 "
param+="min_learning_rate=0.000005 "
param+="save_steps=5000 "
param+="eval_steps=1000 "
param+="scale_loss=1024 "
param+="sharding_parallel_config=split_param,enable_stage1_overlap,enable_stage1_allgather_overlap "


cd ./tests
bash ./test_tipc/dygraph/hybrid_parallelism/qwen/benchmark_common/prepare.sh

bash -c "${param} bash ./test_tipc/dygraph/hybrid_parallelism/qwen/benchmark_common/run_benchmark.sh"
