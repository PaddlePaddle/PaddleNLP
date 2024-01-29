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

unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
unset CUDA_VISIBLE_DEVICES

unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT

export PADDLE_JOB_ID=zzzzssxx

# 启动方式
# export PADDLE_MASTER=10.174.137.30:8678
# export PADDLE_NNODES=2  # 机器数量

python=python3.10

# cd ../model_zoo/gpt-3/external_ops/ &&  ${python} setup.py install && cd -

${python} -m paddle.distributed.launch \
    --ips=10.174.137.30,10.174.137.140 \
	--log_dir /root/paddle_log \
	--gpus 0,1,2,3,4,5,6,7 \
	run_pretrain.py \
    "llama/pretrain-llama_13b-pp4tp2sd2_stage1.json"

# llama/pretrain-llama_13b-tp2sd4_stage2.json 
# llama/pretrain-llama_13b-pp4tp2sd2_stage1.json
# llama/pretrain-llama_13b-tp2sd4_stage2.json 
