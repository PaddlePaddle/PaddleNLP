#!/usr/bin/bash

export GLOG_v=0
export GLOG_logtostderr=1
export PYTHONIOENCODING=utf8
export LC_ALL=C.UTF-8

# PaddlePaddle environment variables
export FLAGS_allocator_strategy=auto_growth
export FLAGS_dynamic_static_unified_comm=0
export FLAGS_use_xqa_optim=1
export FLAGS_gemm_use_half_precision_compute_type=0
export NVIDIA_TF32_OVERRIDE=0

# Model hyperparameters
export MP_NUM=${MP_NUM:-"1"}                                # Number of GPUs
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}    # GPU ids
export MAX_SEQ_LEN=${MAX_SEQ_LEN:-"8192"}
export MAX_DEC_LEN=${MAX_DEC_LEN:-"2048"}
export BATCH_SIZE=${BATCH_SIZE:-"20"}
export BLOCK_BS=${BLOCK_BS:-"4"}
export BLOCK_SIZE=${BLOCK_SIZE:-"64"}
export DTYPE=${DTYPE:-"bfloat16"}
export USE_CACHE_KV_INT8=${USE_CACHE_KV_INT8:-"0"}  # c8 model requires configuration 1
export BLOCK_RATIO=${BLOCK_RATIO:-"0.75"}
export ENC_DEC_BLOCK_NUM=${ENC_DEC_BLOCK_NUM:-"4"}
export FIRST_TOKEN_ID=${FIRST_TOKEN_ID:-"1"}
export MAX_PREFILL_BATCH=${MAX_PREFILL_BATCH:-"4"}
export STOP_THRESHOLD=${STOP_THRESHOLD:-"0"}
export MODEL_DIR=${MODEL_DIR:-"/models"}
export DISTRIBUTED_CONFIG=${DISTRIBUTED_CONFIG:-"${MODEL_DIR}/rank_mapping.csv"}
export CONFIG_JSON_FILE=${CONFIG_JSON_FILE:-"config.json"}
export PUSH_MODE_HTTP_WORKERS=${PUSH_MODE_HTTP_WORKERS:-"4"}

# serving port
export HTTP_PORT=${HTTP_PORT:-"8110"}
export GRPC_PORT=${GRPC_PORT:-"8811"}
export METRICS_PORT=${METRICS_PORT:-"8722"}
export INFER_QUEUE_PORT=${INFER_QUEUE_PORT:-"8813"}
export PUSH_MODE_HTTP_PORT=${PUSH_MODE_HTTP_PORT:-"9965"}

mkdir -p log
rm -rf console.log log/*
rm -rf /dev/shm/*

echo "start serving ..."

tritonserver --exit-timeout-secs 100 --cuda-memory-pool-byte-size 0:0 --cuda-memory-pool-byte-size 1:0 \
                 --cuda-memory-pool-byte-size 2:0 --cuda-memory-pool-byte-size 3:0 --cuda-memory-pool-byte-size 4:0 \
                 --cuda-memory-pool-byte-size 5:0 --cuda-memory-pool-byte-size 6:0 --cuda-memory-pool-byte-size 7:0 \
                 --pinned-memory-pool-byte-size 0 --model-repository llm_model/ \
                 --allow-http false \
                 --grpc-port=${GRPC_PORT} \
                 --metrics-port=${METRICS_PORT} \
                 --log-file log/server.log --log-info true  > log/console.log 2>&1 &

echo "The logs for the model service, please check" ${PWD}"/log/server.log and "${PWD}"/log/workerlog.0"
