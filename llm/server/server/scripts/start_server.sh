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

export GLOG_v=0
export GLOG_logtostderr=1
export PYTHONIOENCODING=utf8
export LC_ALL=C.UTF-8

# PaddlePaddle environment variables
export FLAGS_gemm_use_half_precision_compute_type=0
export NVIDIA_TF32_OVERRIDE=0

# Model hyperparameters
export MP_NUM=${MP_NUM:-"1"}                                # number of model parallelism
export MP_NNODE=${MP_NNODE:-"1"}                            # number of nodes
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}    # GPU ids
export MAX_SEQ_LEN=${MAX_SEQ_LEN:-"8192"}
export MAX_DEC_LEN=${MAX_DEC_LEN:-"8192"}
export BATCH_SIZE=${BATCH_SIZE:-"20"}
export BLOCK_BS=${BLOCK_BS:-"4"}
export BLOCK_RATIO=${BLOCK_RATIO:-"0.75"}
export ENC_DEC_BLOCK_NUM=${ENC_DEC_BLOCK_NUM:-"4"}
export MAX_PREFILL_BATCH=${MAX_PREFILL_BATCH:-"4"}
export STOP_THRESHOLD=${STOP_THRESHOLD:-"0"}

export tag=${tag:-"3.0.0.b4"}
export model_name=$1
export MODEL_DIR=${MODEL_DIR:-"/models"}

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



if [ ! -d "llm_model" ];then
    ln -s /opt/source/PaddleNLP/llm/server/server/llm_model llm_model
fi

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
eval tritonserver --exit-timeout-secs 100000 --cuda-memory-pool-byte-size 0:0 --cuda-memory-pool-byte-size 1:0 \
                 --cuda-memory-pool-byte-size 2:0 --cuda-memory-pool-byte-size 3:0 --cuda-memory-pool-byte-size 4:0 \
                 --cuda-memory-pool-byte-size 5:0 --cuda-memory-pool-byte-size 6:0 --cuda-memory-pool-byte-size 7:0 \
                 --pinned-memory-pool-byte-size 0 --model-repository llm_model/ \
                 --allow-http false \
                 --grpc-port=${SERVICE_GRPC_PORT} \
                 --metrics-port=${METRICS_HTTP_PORT} \
                 --log-file log/server.log --log-info true $LOG_REDIRECT &

echo "The logs for the model service, please check" ${PWD}"/log/server.log and "${PWD}"/log/workerlog.0"
