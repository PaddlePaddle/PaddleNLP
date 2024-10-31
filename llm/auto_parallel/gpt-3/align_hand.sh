# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT

export NNODES=1
export PADDLE_TRAINERS_NUM=1

export GLOG_v=0

export FLAGS_cudnn_deterministic=1
export FLAGS_embedding_deterministic=1 
# export FLAGS_max_inplace_grad_add=65536
export FLAGS_enable_auto_parallel_align_mode=1
export test_ci_no_save_model=0

set -x
unset CUDA_VISIBLE_DEVICES

task_name="gpt3_13b_hand"
log_dir="log/$task_name"
rm -rf $log_dir

# export PYTHONPATH=../../../:$PYTHONPATH

python -u -m paddle.distributed.launch \
    --gpus "6,7" \
    --log_dir ${log_dir} \
    /root/paddlejob/workspace/env_run/zhangwl/zwl/PaddleNLP/llm/run_pretrain.py \
    --model_name_or_path test_config.json \
    --tokenizer_name_or_path gpt3-13B-en \
    --input_dir "./data" \
    --output_dir "output/$task_name" \
    --split 949,50,1 \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --scale_loss 1024 \
    --learning_rate 0.00001 \
    --min_learning_rate 0.000001 \
    --max_steps 10000 \
    --save_steps 1000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --logging_steps 1 \
    --continue_training 0 \
    --dataloader_num_workers 4 \
    --eval_steps 100000 \
    --report_to "visualdl" \
    --disable_tqdm true \
    --do_train \
    --do_eval \
    --device "gpu" \
    --sharding "stage1" \
    --sharding_parallel_degree 1 \
    --tensor_parallel_degree 2 \
    --pipeline_parallel_degree 1 \
    --sequence_parallel 0 \
    --use_flash_attention 1 \
    --use_fast_layer_norm 0 \
    --fuse_attention_qkv 1 \
    --use_fused_dropout_add 0 \
    --use_fused_linear 0 \
    --enable_linear_fused_grad_add 0 \
    --recompute 0 \
    --recompute_use_reentrant true \
    --recompute_granularity "full" \
    --pp_recompute_interval 1 \
    --gradient_accumulation_steps 32 \
    --max_grad_norm 0 \
    --bf16 1 \
    --fp16_opt_level "O2"  \
    --amp_master_grad true \
    --attention_probs_dropout_prob 0 \
    --hidden_dropout_prob 0 \
    --save_sharded_model false \
    --sharding_parallel_config "enable_stage1_tensor_fusion enable_stage1_overlap" \
    --tensor_parallel_config "enable_mp_async_allreduce" \
    --resume_from_checkpoint "output/gpt3_13b_hand/checkpoint-1"
    # --sharding_parallel_config "enable_stage1_tensor_fusion enable_stage1_overlap" \
    # --tensor_parallel_config "enable_mp_async_allreduce enable_mp_skip_c_identity enable_mp_fused_linear_param_grad_add" \
    # --pipeline_parallel_config "enable_sharding_comm_overlap" \