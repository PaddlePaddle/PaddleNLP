# setup virtual env
# source /home/full_auto/venv/bin/activate

# setup paddlenlp env
# PADDLENLP_PATH="/home/full_auto/PaddleNLP"
# export PYTHONPATH=${PADDLENLP_PATH}:$PYTHONPATH 

# setup nccl env
# export FLAGS_nccl_dir=/opt/nccl2.15.5/usr/lib/x86_64-linux-gnu
unset NCCL_DEBUG_FILE
unset NCCL_DEBUG_SUBSYS
unset NCCL_ERROR_FILE
unset NCCL_IB_CONNECT_RETRY_CNT
unset NCCL_IB_CUDA_SUPPORT
unset NCCL_IB_DISABLE
unset NCCL_IB_QPS_PER_CONNECTION
unset NCCL_IB_TIMEOUT
unset NCCL_P2P_DISABLE
unset NCCL_SOCKET_IFNAME
unset NCCL_VERSION

unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
unset PADDLE_TRAINERS_NUM
unset PADDLE_TRAINER_ID

export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG="INFO"
export log_dir=log_new

rm -rf $log_dir


python -m paddle.distributed.launch --log_dir $log_dir --devices "0,1,2,3,4,5,6,7" \
    --auto_cluster_config true \
    --master=10.127.24.147:8091 \
    --nnodes=3 ./tools/auto.py \
    -c ./ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_full_auto_parallel_n3.yaml

    

    