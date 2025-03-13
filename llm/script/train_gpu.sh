# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

nnodes=$PADDLE_TRAINERS_NUM
rank=$PADDLE_TRAINER_ID

for name in `env | grep -E 'PADDLE|ENDPOINT' | awk -F'=' '{print $1}'`; do
  unset ${name}
done

#export FLAGS_shard_bypass_dygraph_optimizer=1
export NCCL_IB_GID_INDEX=3
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_IB_TRAFFIC_CLASS=162

#export NVSHMEM_IB_ENABLE_IBGDA=true
##export NVSHMEM_DISABLE_P2P=1
export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME==xgbe0

START_RANK=0
END_RANK=8

if [[ $rank -lt $START_RANK ]]; then
    exit 0
fi

if [[ $rank -ge $END_RANK ]]; then
    exit 0
fi

rank=$(($rank-$START_RANK))
nnodes=$(($END_RANK-$START_RANK))

master=`cat /root/paddlejob/workspace/hostfile | head -n 1 | awk '{print $1}'`
port=36679
export PYTHONPATH=../:$PYTHONPATH
export PATH=/opt/nvidia/nsight-systems/2025.1.1/bin/:$PATH

python3.10 -m paddle.distributed.launch \
    --log_dir output/paddle_distributed_logs \
    --master $master:$port \
    --nnodes $nnodes \
    --rank $rank \
    --run_mode=collective \
    ${script:-run_pretrain.py}  \
    $@
