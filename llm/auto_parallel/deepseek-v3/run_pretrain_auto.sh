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

#!/bin/bash
set -x
unset CUDA_VISIBLE_DEVICES

task_name="deepseekv3"
rm -rf output/$task_name/
rm -rf "output/$task_name""_log"

export SOT_LOG_LEVEL=4
export PYTHONPATH=../../../:$PYTHONPATH


#ulimit -c unlimited
# export GLOG_v=3

# export FLAGS_call_stack_level=3
# export FLAGS_use_cuda_managed_memory=true

# export FLAGS_embedding_deterministic=1        
# export FLAGS_cudnn_deterministic=1
# export NVIDIA_TF32_OVERRIDE=0

export FLAGS_enable_moe_utils=true
export FLAGS_call_stack_level=3

to_static=0  # 是否开启动转静训练

python -u  -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir  "output/$task_name""_log" \
    run_pretrain_auto.py \
    --model_type "deepseekv3_auto" \
    --model_name_or_path "deepseek-ai/DeepSeek-V3" \
    --tokenizer_name_or_path "deepseek-ai/DeepSeek-V3" \
    --input_dir "./data" \
    --output_dir "output/$task_name" \
    --split 949,50,1 \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --fuse_attention_ffn true \
    --fuse_attention_qkv true \
    --fuse_sequence_parallel_allreduce true \
    --use_flash_attention true \
    --use_fused_rope true \
    --use_fused_rms_norm true \
    --bf16 True \
    --fp16_opt_level "O2"  \
    --scale_loss 1024 \
    --pipeline_parallel_degree 1 \
    --tensor_parallel_degree 1 \
    --sharding_parallel_degree 8 \
    --sharding "stage1" \
    --learning_rate 0.0001 \
    --min_learning_rate 0.00001 \
    --max_steps 2000 \
    --moe_group "dp" \
    --save_steps 100000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --logging_steps 1\
    --dataloader_num_workers 1 \
    --eval_steps 1000000 \
    --disable_tqdm true \
    --continue_training 0\
    --recompute 0 \
    --do_train \
    --do_eval \
    --device "gpu" \
    --data_impl "mmap" \
    --enable_auto_parallel 1 \
    --max_grad_norm 1.0 \
    --num_hidden_layers 2 \
    --first_k_dense_replace 0 \
    --n_routed_experts 16 \
    --use_intermediate_api true \
    --to_static $to_static \
