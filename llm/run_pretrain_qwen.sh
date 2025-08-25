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

# 环境变量（保持与项目一致的最小必要项）
export PYTHONPATH=../:$PYTHONPATH
export FLAGS_call_stack_level=3
export NVIDIA_TF32_OVERRIDE=0
export FLAGS_cudnn_deterministic=True
export CUDA_DEVICE_MAX_CONNECTIONS=1
export FLAGS_embedding_deterministic=1

# 输出与日志目录
case_out_dir="./qwen3_moe_outputs/model_output"
case_log_dir="./qwen3_moe_outputs/qwen3_pretrain_dy_log"

# 清理旧目录（可按需注释掉）
rm -rf "$case_out_dir"
rm -rf "$case_log_dir"

# 启动训练（参数严格对齐 JSON）
# --model_name_or_path "Qwen/Qwen3-30B-A3B-Instruct-2507" \
# --tokenizer_name_or_path "Qwen/Qwen3-30B-A3B-Instruct-2507" \
python -u -m paddle.distributed.launch \
    --gpus "0,1" \
    --log_dir "$case_log_dir" \
    run_pretrain.py \
    --model_name_or_path "./qwen3-moe" \
    --tokenizer_name_or_path "./qwen3-moe" \
    --input_dir "./data" \
    --split "949,50,1" \
    --max_seq_length 256 \
    --num_hidden_layers 4 \
    --output_dir "$case_out_dir" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 8 \
    --tensor_parallel_degree 2 \
    --pipeline_parallel_degree 1 \
    --tensor_parallel_config "enable_delay_scale_loss enable_mp_async_allreduce enable_mp_skip_c_identity" \
    --pipeline_parallel_config "enable_delay_scale_loss enable_release_grads disable_partial_send_recv enable_overlap_p2p_comm" \
    --virtual_pp_degree 1 \
    --sequence_parallel 0 \
    --use_flash_attention 0 \
    --use_fused_rms_norm 0 \
    --enable_linear_fused_grad_add 0 \
    --learning_rate 3e-05 \
    --min_learning_rate 3e-06 \
    --warmup_steps 30 \
    --logging_steps 1 \
    --max_steps 3 \
    --save_steps 3 \
    --eval_steps 1000 \
    --weight_decay 0.01 \
    --fp16 1 \
    --fp16_opt_level "O2" \
    --amp_master_grad 1 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --dataloader_num_workers 1 \
    --continue_training 0 \
    --do_train true \
    --do_eval false \
    --do_predict false \
    --disable_tqdm true \
    --skip_profile_timer true \
    --recompute 0 \
    --save_total_limit 2 \
    --device "gpu" \
    --save_sharded_model 0 \
    --unified_checkpoint 0 \
    --using_flex_checkpoint 1 \
    --fuse_attention_qkv true \
    --fuse_attention_ffn true \
    # --resume_from_checkpoint "$case_out_dir/checkpoint-3" \
    # --aoa_config '{
    # "aoa_statements": [
    #     "llama.layers.$LAYER_ID.self_attn.q_proj.weight, llama.layers.$LAYER_ID.self_attn.k_proj.weight, llama.layers.$LAYER_ID.self_attn.v_proj.weight -> llama.layers.$LAYER_ID.self_attn.qkv_proj.weight ,axis=1",
    #     "llama.layers.$LAYER_ID.self_attn.q_proj.weight.w_0, llama.layers.$LAYER_ID.self_attn.k_proj.weight.w_0, llama.layers.$LAYER_ID.self_attn.v_proj.weight.w_0 -> llama.layers.$LAYER_ID.self_attn.qkv_proj.weight.w_0  ,axis=1",
    #     "llama.layers.$LAYER_ID.self_attn.q_proj.weight.moment1_0, llama.layers.$LAYER_ID.self_attn.k_proj.weight.moment1_0, llama.layers.$LAYER_ID.self_attn.v_proj.weight.moment1_0 -> llama.layers.$LAYER_ID.self_attn.qkv_proj.weight.moment1_0  ,axis=1",
    #     "llama.layers.$LAYER_ID.self_attn.q_proj.weight.moment2_0, llama.layers.$LAYER_ID.self_attn.k_proj.weight.moment2_0, llama.layers.$LAYER_ID.self_attn.v_proj.weight.moment2_0 -> llama.layers.$LAYER_ID.self_attn.qkv_proj.weight.moment2_0  ,axis=1",
    #     "llama.layers.$LAYER_ID.self_attn.q_proj.weight.beta1_pow_acc_0, llama.layers.$LAYER_ID.self_attn.k_proj.weight.beta1_pow_acc_0, llama.layers.$LAYER_ID.self_attn.v_proj.weight.beta1_pow_acc_0 -> llama.layers.$LAYER_ID.self_attn.qkv_proj.weight.beta1_pow_acc_0  ,axis=0",
    #     "llama.layers.$LAYER_ID.self_attn.q_proj.weight.beta2_pow_acc_0, llama.layers.$LAYER_ID.self_attn.k_proj.weight.beta2_pow_acc_0, llama.layers.$LAYER_ID.self_attn.v_proj.weight.beta2_pow_acc_0 -> llama.layers.$LAYER_ID.self_attn.qkv_proj.weight.beta2_pow_acc_0  ,axis=0"
    # ]
    # }'\
    # --sharding "stage1" \
    # --sharding_parallel_degree 2 \
    # --fuse_attention_ffn true \
    
