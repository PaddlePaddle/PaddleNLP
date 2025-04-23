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

WORK_ROOT=/work/models
export PYTHONPATH=${WORK_ROOT}/PaddleNLP:${WORK_ROOT}/PaPerf:$PYTHONPATH

export CUDA_VISIBLE_DEVICES="0,1,2,3"

unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT

nnodes=1 #$PADDLE_TRAINERS_NUM
rank=0 #$PADDLE_TRAINER_ID

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
END_RANK=1

if [[ $rank -lt $START_RANK ]]; then
    exit 0
fi

if [[ $rank -ge $END_RANK ]]; then
    exit 0
fi

rank=$(($rank-$START_RANK))
nnodes=$(($END_RANK-$START_RANK))

export FLAGS_enable_ap=1
export AP_WORKSPACE_DIR=$(pwd)/ap_workspace
export AP_PATH=/work/abstract_pass/Athena/tests/ap

export FLAGS_check_infer_symbolic=1
export FLAGS_enable_pir_api=1
export FLAGS_cinn_bucket_compile=True
export FLAGS_prim_enable_dynamic=true
export FLAGS_prim_all=True
export FLAGS_pir_apply_shape_optimization_pass=1
export FLAGS_group_schedule_tiling_first=1
export FLAGS_cinn_new_group_scheduler=1
export FLAGS_cinn_enable_vectorize=true

#export GLOG_vmodule=*=4
#export GLOG_vmodule=naive_dl_handler=4,ap_generic_drr_pass=6

#master=`cat /root/paddlejob/workspace/hostfile | head -n 1 | awk '{print $1}'`
#port=36679
export PYTHONPATH=../:$PYTHONPATH
export PATH=/opt/nvidia/nsight-systems/2025.1.1/bin/:$PATH
#nsys_args="nsys profile --stats true -w true -t cuda,nvtx --capture-range=cudaProfilerApi -x true --force-overwrite true -o deepseek-v3.bf16"

rm -rf checkpoints log ap_workspace/*

${nsys_args} python -m paddle.distributed.launch \
    --nnodes 1 --nproc_per_node 4 --log_dir log \
    ../run_pretrain.py  \
    ../config/deepseek-v3/pretrain_argument.json 2>&1 | tee log_deepseekv3.bf16.txt
