#!/bin/bash

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

task_name_or_path="llama2-13b-auto"

#export XPUAPI_DEBUG=0x1
#export XPURT_DISPATCH_MODE=PROFILING
export XBLAS_FC_HBM_VERSION=40

# PaddlePaddle
export FLAGS_use_stride_kernel="0"
export XPU_PADDLE_L3_SIZE=98566144 # 94 MB
export XPU_CDNN_CLUSTER_PARALLEL=1
export XPU_CDNN_CLUSTER_PARALLEL_STREAM_NUMBER=2

# PDC
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
unset PADDLE_TRAINERS_NUM

# BKCL
# export BKCL_DEBUG=1
# Multi-computer RDMA
#export BKCL_ENABLE_XDR=1
#export BKCL_RDMA_FORCE_TREE=1
#export BKCL_TREE_THRESHOLD=0
#export BKCL_RDMA_NICS=xgbe1,xgbe1,xgbe2,xgbe2,xgbe3,xgbe3,xgbe4,xgbe4
#export BKCL_SOCKET_IFNAME=xgbe0
#export BKCL_FORCE_L3_RDMA=0
export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib64
echo "bkcl version:"
strings ${bkcl_location}/libbkcl.so | grep COM

export CUDA_DEVICE_MAX_CONNECTIONS=8

#PYTHONPATH
export PYTHONPATH=../../../:$PYTHONPATH

# for debug
#export GLOG_v=10
export FLAGS_call_stack_level=2

rm -rf output/$task_name_or_path
PYTHONPATH=../:$PYTHONPATH  \
python -u  -m paddle.distributed.launch \
    --xpus "0,1,2,3,4,5,6,7" \
    --log_dir "output/$task_name_or_path/" \
    run_pretrain_auto.py \
    --model_name_or_path "meta-llama/Llama-2-13b" \
    --tokenizer_name_or_path "meta-llama/Llama-2-13b" \
    --input_dir "./data" \
    --output_dir "output/$task_name_or_path" \
    --split 949,50,1 \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --use_flash_attention 1 \
    --use_fused_rope 1 \
    --fuse_attention_ffn 1 \
    --fuse_attention_qkv 1 \
    --use_fused_rms_norm 0 \
    --num_hidden_layers 4 \
    --bf16 \
    --fp16_opt_level "O2"  \
    --amp_master_grad true \
    --scale_loss 1024 \
    --learning_rate 0.00003 \
    --min_learning_rate 0.000005 \
    --lr_scheduler_type "cosine" \
    --max_steps 10 \
    --save_steps 100000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --sequence_parallel 0 \
    --dataloader_num_workers 4 \
    --pipeline_parallel_degree 1 \
    --tensor_parallel_degree 1 \
    --gradient_accumulation_steps 1 \
    --eval_steps 1000 \
    --report_to "visualdl" \
    --disable_tqdm true \
    --continue_training 0 \
    --recompute 0 \
    --do_train \
    --seed 1026 \
    --device "xpu" \
    --enable_auto_parallel 1 \
    --to_static 1
