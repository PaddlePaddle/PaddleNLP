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

set -ex

# export LOG_LEVEL_ALL=0
export HABANA_LOGS=./logs

# export HCCL_COMM_ID=127.0.0.1:5555
# export INTEL_HPU_VISIBLE_DEVICES=0,1 # 3,4
export INTEL_HPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PADDLE_DISTRI_BACKEND=xccl
export PADDLE_XCCL_BACKEND=intel_hpu
# PYTHONPATH=../../../:$PYTHONPATH  \
export FLAGS_intel_hpu_runtime_debug=0

# export HABANA_PROFILE=1
# export HABANA_PROFILE_WRITE_HLTV_WITH_HOST=1

echo $INTEL_HPU_VISIBLE_DEVICES

# export GRAPH_VISUALIZATION=1
# export ENABLE_EXPERIMENTAL_FLAGS=1
# export VISUALIZATION_MODE=0

#GLOG_v=10
python -m paddle.distributed.launch --devices "3,5" test_llama.py 2>&1 | tee test_llama_2x.log


