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

WORK_ROOT=/root/paddlejob/workspace/env_run/liuyiqun
export PYTHONPATH=${WORK_ROOT}/env/virtualenvs_cuda12.8/paddle_py310_yiqun
export PATH=${PYTHONPATH}/bin:${PATH}

#pip install -r $WORK_ROOT/PaddleNLP/requirements-dev.txt 
#pip install -r $WORK_ROOT/PaddleNLP/requirements.txt 
#pip install colorlog
#pip install ml_dtypes
#pip install safetensors
#pip install aistudio_sdk
#pip install kitchen
#exit

export PYTHONPATH=${WORK_ROOT}/PaddleNLP:${WORK_ROOT}/PaPerf:$PYTHONPATH

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

if [[ ${rank} -lt $START_RANK ]]; then
    exit 0
fi

if [[ ${rank} -ge $END_RANK ]]; then
    exit 0
fi

rank=$(($rank - $START_RANK))
nnodes=$(($END_RANK - $START_RANK))
echo "rank: ${rank}, nnodes: ${nnodes}"

python -c "import paddle; print(paddle.version.commit)"

master=`cat /root/paddlejob/workspace/hostfile | head -n 1 | awk '{print $1}'`
port=36699
export PYTHONPATH=../:$PYTHONPATH
export PATH=/opt/nvidia/nsight-systems/2025.1.1/bin/:$PATH

SUFFIX=${nnodes}nodes_timer_20250320_0

rm -rf checkpoints log_${nnodes}nodes${SUFFIX}

python -m paddle.distributed.launch \
    --log_dir log_${SUFFIX} \
    --master $master:$port \
    --nnodes $nnodes \
    --rank $rank \
    --run_mode=collective \
    ../run_pretrain.py  \
    ../config/deepseek-v3/pretrain_argument.json 2>&1 | tee log_deepseek-v3.bf16.${SUFFIX}.txt 
