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

# export CUDA_LAUNCH_BLOCKING=1
# unset http_proxy && unset https_proxy
# pip install -U --force-reinstall /root/paddlejob/workspace/env_run/output/paddlepaddle_gpu-0.0.0-cp310-cp310-linux_x86_64.whl

set -ex

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

START_RANK=84
END_RANK=85

if [[ $rank -lt $START_RANK ]]; then
    exit 0
fi

if [[ $rank -ge $END_RANK ]]; then
    exit 0
fi


skip_kill_time=${1:-"False"}
function kill_impl() {
    skip_kill_time=$1
    # kill aadiff test finally.
    pids=`ps -ef | grep pretrain.py | grep -v grep | awk '{print $2}'`
    if [[ "$pids" != "" ]] ; then
        echo $pids
        echo $pids | xargs kill -9
    fi

    echo "Killing processes on gpu"
    lsof /dev/nvidia* | awk '{print $2}' | xargs -I {} kill -9 {}
}

kill_impl $skip_kill_time || true

nvidia-smi |grep MiB


rank=$(($rank-$START_RANK))
nnodes=$(($END_RANK-$START_RANK))

# master=`cat /root/paddlejob/workspace/hostfile | head -n 1 | awk '{print $1}'`
master=`cat /root/paddlejob/workspace/hostfile | head -n $(($START_RANK+1)) | tail -n 1 | awk '{print $1}'`
port=36679



source /root/paddlejob/workspace/env_run/liuhongyu/py310/bin/activate
PYTHONPATH=/root/paddlejob/workspace/env_run/liuhongyu/new_env/PaddleNLP/:$PYTHONPATH


#export PATH=/opt/nvidia/nsight-systems/2025.1.1/bin/:$PATH

mkdir -p output/paddle_distributed_logs

# /opt/nvidia/nsight-systems/2025.1.1/bin/nsys profile --stats=true -t cuda,nvtx -o pp4_ep64 --capture-range=cudaProfilerApi --force-overwrite true \
# nsys profile -t cuda,nvtx -o pp4_ep64_mask_gemm_duilpipe_overlap_acc8 --force-overwrite true \

which python

export CUDA_PATH=/usr/local/cuda-12.9/
export DSV3_USE_FP8_GEMM=True
export DSV3_USE_ATTEN_RECOMPUTE=True
export FA_VERSION=3
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0

#/root/paddlejob/workspace/env_run/liuhongyu/install/nsys/bin/nsys profile --stats true -w true -t cuda,nvtx \
python3 -m paddle.distributed.launch \
    --log_dir output/paddle_distributed_logs \
    --master $master:$port \
    --nnodes $nnodes \
    --rank $rank \
    --run_mode=collective \
    run_pretrain.py  \
    config/deepseek-v3/pretrain_argument.json
