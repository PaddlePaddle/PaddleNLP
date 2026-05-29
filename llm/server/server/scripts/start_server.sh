#!/usr/bin/bash

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

export PYTHONPATH=/root/paddlejob/workspace/env_run/output/changwenbin/PaddleNLP
export PYTHONPATH=/root/paddlejob/workspace/env_run/output/changwenbin/PaddleNLP/llm/server/server/:$PYTHONPATH
export PYTHONPATH=/root/paddlejob/workspace/env_run/output/changwenbin/PaddleNLP/llm/:$PYTHONPATH
export PATH=/opt/tritonserver/bin/:$PATH

export FLAGS_cascade_attention_max_partition_size=163840
export FLAGS_mla_use_tensorcore=0
export USE_DYNAMIC_GRAPH=1

export GLOG_v=0
export GLOG_logtostderr=1
export PYTHONIOENCODING=utf8
export LC_ALL=C.UTF-8

# PaddlePaddle environment variables
export FLAGS_gemm_use_half_precision_compute_type=0
export NVIDIA_TF32_OVERRIDE=0

export NCCL_ALGO=Tree
export FLAGS_use_wintx_gemm=True


# Model hyperparameters
export MP_NUM=${MP_NUM:-"4"}                                # number of model parallelism
export MP_NNODE=${MP_NNODE:-"1"}                            # number of nodes
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}    # GPU ids
export MAX_SEQ_LEN=${MAX_SEQ_LEN:-"8192"}
export MAX_DEC_LEN=${MAX_DEC_LEN:-"4096"}
export BATCH_SIZE=${BATCH_SIZE:-"128"}
export BLOCK_BS=${BLOCK_BS:-"1"}
export BLOCK_RATIO=${BLOCK_RATIO:-"0.75"}
export ENC_DEC_BLOCK_NUM=${ENC_DEC_BLOCK_NUM:-"4"}
export MAX_PREFILL_BATCH=${MAX_PREFILL_BATCH:-"4"}
export STOP_THRESHOLD=${STOP_THRESHOLD:-"0"}

export tag=${tag:-"3.0.0.b4"}
export model_name=$1
export MODEL_DIR=/root/paddlejob/workspace/env_run/output/model/dsv3_w2_tq2.75_w4_channel_tp4_new
# export MODEL_DIR=/root/paddlejob/workspace/env_run/output/bos_model/community/deepseek-ai/DeepSeek-V3-0324
# export MODEL_DIR=/root/.paddlenlp/models/Qwen/Qwen1.5-MoE-A2.7B-Chat
# export MODEL_DIR=/root/paddlejob/workspace/env_run/output/changwenbin/PaddleNLP/llm/predict/static/Qwen/Qwen1.5-MoE-A2.7B-Chat
# ${MODEL_DIR:-"/models"}

if [ ! "$model_name" == "" ]; then
    export MODEL_DIR=${MODEL_DIR}/${model_name}
    mkdir -p $MODEL_DIR
fi

export CONFIG_JSON_FILE=${CONFIG_JSON_FILE:-"config.json"}
export PUSH_MODE_HTTP_WORKERS=${PUSH_MODE_HTTP_WORKERS:-"4"}

# serving port
export HEALTH_HTTP_PORT=${HTTP_PORT:-${HEALTH_HTTP_PORT:-"8110"}}
export METRICS_HTTP_PORT=${METRICS_PORT:-${METRICS_HTTP_PORT:-"8722"}}
export SERVICE_GRPC_PORT=${GRPC_PORT:-${SERVICE_GRPC_PORT:-"8811"}}
export INTER_PROC_PORT=${INTER_QUEUE_PORT:-${INTER_PROC_PORT:-"8813"}}
export SERVICE_HTTP_PORT=${PUSH_MODE_HTTP_PORT:-${SERVICE_HTTP_PORT:-"9965"}}

check_port_occupied() {
    local port=$1
    if netstat -tuln | grep -q ":${port}\b"; then
        echo  "PORT: ${port} occupied! Please change the port!"
        exit 1
    fi
}

check_port_occupied ${HEALTH_HTTP_PORT}
check_port_occupied ${METRICS_HTTP_PORT}
check_port_occupied ${SERVICE_GRPC_PORT}
check_port_occupied ${INTER_PROC_PORT}
check_port_occupied ${SERVICE_HTTP_PORT}



# if [ ! -d "llm_model" ];then
#     ln -s /opt/source/PaddleNLP/llm/server/server/llm_model llm_model
# fi

mkdir -p log
rm -rf console.log log/*
rm -rf /dev/shm/*

FED_POD_IP=$(hostname -i)
if [ "$MP_NNODE" -gt 1 ]; then
    POD_0_IP=$POD_0_IP
    export HOST_IP=$FED_POD_IP
else
    POD_0_IP="127.0.0.1"
    HOST_IP="127.0.0.1"
    # 屏蔽平台预设的环境变量，因为框架采用兼容升级，检测到这些配置会使用原方式启动
    # unset PADDLE_ELASTIC_JOB_ID
    # unset PADDLE_TRAINER_ENDPOINTS
    # unset DISTRIBUTED_TRAINER_ENDPOINTS
    # unset FLAGS_START_PORT
    # unset PADDLE_ELASTIC_TIMEOUT
    # nnodes=$PADDLE_TRAINERS_NUM
    # rank=$PADDLE_TRAINER_ID

    # for name in `env | grep -E 'PADDLE|ENDPOINT' | awk -F'=' '{print $1}'`; do
    # unset ${name}
    # done

    # START_RANK=0
    # END_RANK=$nnodes
    # END_RANK=1

    # if [[ $rank -lt $START_RANK ]]; then
    #     echo "rank exit"
    #     exit 0
    # fi

    # if [[ $rank -ge $END_RANK ]]; then
    #     echo "rank exit"
    #     exit 0
    # fi

    # rank=$(($rank-$START_RANK))
    # nnodes=$(($END_RANK-$START_RANK))
    # master=`cat /root/paddlejob/workspace/hostfile | head -n $(($START_RANK+1)) | tail -n 1 | awk '{print $1}'`
    # port=36677

    # set -ex
fi

echo "POD_0_IP: $POD_0_IP HOST_IP: $HOST_IP"

if [ "$POD_0_IP" == "$HOST_IP" ]; then
    echo "Master node, start serving ..."
else
    echo "Slave node, start push mode"
    # waiting for master node to start serving ...
    sleep ${SERVER_WAITTING_TIME:-"25"}
fi

OUTPUT_LOG_TO_CONSOLE=${OUTPUT_LOG_TO_CONSOLE:-"0"}
# Set the log redirection based on whether logs should be output to the console
LOG_REDIRECT=""
# If OUTPUT_LOG_TO_CONSOLE is set to "1", redirect logs to the console log file
if [ "$OUTPUT_LOG_TO_CONSOLE" == "1" ]; then
    LOG_REDIRECT="> log/console.log 2>&1"
fi
eval nohup fastdeployserver --exit-timeout-secs 100000 --cuda-memory-pool-byte-size 0:0 --cuda-memory-pool-byte-size 1:0 \
                 --cuda-memory-pool-byte-size 2:0 --cuda-memory-pool-byte-size 3:0 --cuda-memory-pool-byte-size 4:0 \
                 --cuda-memory-pool-byte-size 5:0 --cuda-memory-pool-byte-size 6:0 --cuda-memory-pool-byte-size 7:0 \
                 --pinned-memory-pool-byte-size 0 --model-repository /root/paddlejob/workspace/env_run/output/changwenbin/PaddleNLP/llm/server/server/llm_model/ \
                 --allow-http false \
                 --grpc-port=${SERVICE_GRPC_PORT} \
                 --metrics-port=${METRICS_HTTP_PORT} \
                 --log-file log/server.log --log-info true $LOG_REDIRECT &

echo "The logs for the model service, please check" ${PWD}"/log/server.log and "${PWD}"/log/workerlog.0"
