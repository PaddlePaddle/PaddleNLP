#!/bin/bash
set -xe

# Paddle debug envs
export GLOG_v=1
export GLOG_logtostderr=1

# Unset proxy
unset https_proxy http_proxy

# NCCL debug envs
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
# Comment it if your nccl support IB
export NCCL_IB_DISABLE=1

# Add your nodes endpoints here.
export worker_endpoints=127.0.0.1:9184,127.0.0.1:9185
export current_endpoint=127.0.0.1:9184
export CUDA_VISIBLE_DEVICES=0

./train.sh -local n > 0.log 2>&1 &

# Add your nodes endpoints here.
export current_endpoint=127.0.0.1:9185
export CUDA_VISIBLE_DEVICES=1

./train.sh -local n > 1.log 2>&1 &
