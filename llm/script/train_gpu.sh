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
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME==eth0

START_RANK=55
END_RANK=56

if [[ $rank -lt $START_RANK ]]; then
    exit 0
fi

if [[ $rank -ge $END_RANK ]]; then
    exit 0
fi

rank=$(($rank-$START_RANK))
nnodes=$(($END_RANK-$START_RANK))

master=`cat /root/paddlejob/workspace/hostfile | head -n $(($START_RANK+1)) | tail -n 1 | awk '{print $1}'`
port=36679
export PYTHONPATH=../:$PYTHONPATH
export PATH=/opt/nvidia/nsight-systems/2025.1.1/bin/:$PATH

export DSV3_USE_FP8_GEMM=true
export DSV3_USE_ATTEN_RECOMPUTE=true
export FA_VERSION=3
export CUDA_PATH=/usr/local/cuda-12.9
export FLAGS_share_tensor_for_grad_tensor_holder=1
export DSV3_USE_FP8_DISPATCH=False

sh script/kill_process.sh 
source /root/paddlejob/workspace/env_run/zhangbo/env_ds0702/bin/activate

/opt/nvidia/nsight-systems/2025.1.1/bin/nsys profile --stats=true -t cuda,nvtx -o fp8_overlap_quant --force-overwrite true \
python3.10 -m paddle.distributed.launch \
    --log_dir output/paddle_distributed_logs \
    --master $master:$port \
    --nnodes $nnodes \
    --rank $rank \
    --run_mode=collective \
    ${script:-run_pretrain.py}  \
    $@
