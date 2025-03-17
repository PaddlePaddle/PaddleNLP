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
export GLOG_minloglevel=2

rm -rf output/$task_name_or_path
PYTHONPATH=../:$PYTHONPATH  \
python -u  -m paddle.distributed.launch \
    --xpus "0,1,2,3,4,5,6,7" \
    --log_dir "output/$task_name_or_path/" \
    ./run_pretrain_auto.py \
    --model_name_or_path "meta-llama/Llama-2-13b" \
    --tokenizer_name_or_path "meta-llama/Llama-2-13b" \
    --input_dir "./data" \
    --output_dir "output/$task_name_or_path" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 4 \
    --tensor_parallel_degree 1 \
    --pipeline_parallel_degree 4 \
    --sharding "stage1" \
    --data_parallel_config "enable_allreduce_avg_in_gradinent_scale gradient_sync_after_accumulate" \
    --sharding_parallel_config "enable_overlap" \
    --tensor_parallel_config "enable_mp_async_allreduce" \
    --pipeline_parallel_config "enable_send_recv_overlap" \
    --sequence_parallel 0 \
    --use_flash_attention 1 \
    --use_fused_rms_norm 0 \
    --use_fast_layer_norm 1 \
    --fuse_attention_ffn 1 \
    --fuse_attention_qkv 1 \
    --use_fused_rope 1 \
    --enable_linear_fused_grad_add 0 \
    --max_seq_length 4096 \
    --learning_rate 3e-05 \
    --min_learning_rate 3e-06 \
    --warmup_steps 30 \
    --logging_steps 1 \
    --max_steps 1000 \
    --save_steps 100000 \
    --eval_steps 10000 \
    --weight_decay 0.01 \
    --do_train 1 \
    --do_eval 0 \
    --bf16 \
    --fp16_opt_level "O2"  \
    --amp_master_grad true \
    --amp_custom_black_list "reduce_sum" "c_softmax_with_cross_entropy" \
    --amp_custom_white_list "lookup_table" "lookup_table_v2" \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --dataloader_num_workers 1 \
    --continue_training 0 \
    --do_predict 0 \
    --disable_tqdm 1 \
    --skip_profile_timer 1 \
    --recompute 0 \
    --recompute_use_reentrant 1 \
    --distributed_dataloader 0 \
    --recompute_granularity "full" \
    --save_total_limit 2 \
    --device "xpu" \
    --enable_auto_parallel 1 \
    --to_static 0 \
    --hybrid_parallel_topo_order "sharding_first" \
