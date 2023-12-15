# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

set -x
unset CUDA_VISIBLE_DEVICES

rm -rf log
rm -rf output


export PYTHONPATH=../../:$PYTHONPATH
python -u  -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir "./log" \
    --auto_tuner_json ./benchmark/llama13b_pretrain_gbs32_autoconfig.json \
    run_pretrain.py \
    --model_type "llama" \
    --model_name_or_path "facebook/llama-13b" \
    --tokenizer_name_or_path "facebook/llama-13b" \
    --input_dir "./data" \
    --output_dir "./output" \
    --split 949,50,1 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 4 \
    --bf16  \
    --fp16_opt_level "O2"  \
    --amp_master_grad \
    --tensor_parallel_degree 2 \
    --pipeline_parallel_degree 1 \
    --virtual_pp_degree 1 \
    --sharding "stage1" \
    --sharding_parallel_degree 4 \
    --tensor_parallel_config "enable_mp_async_allreduce enable_mp_skip_c_identity enable_mp_fused_linear_param_grad_add" \
    --pipeline_parallel_config "enable_sharding_comm_overlap" \
    --sharding_parallel_config "enable_stage1_tensor_fusion enable_stage1_overlap" \
    --recompute 0 \
    --pp_recompute_interval 1 \
    --recompute_use_reentrant true \
    --learning_rate 0.00001 \
    --min_learning_rate 0.000001 \
    --max_steps 150 \
    --save_steps 50000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --dataloader_num_workers 1 \
    --eval_steps 1001 \
    --disable_tqdm true \
    --continue_training 0 \
    --do_train \
    --device "gpu" \
    --use_flash_attention true \
    --use_fused_rms_norm true \
    --use_fused_rope true \
    --fuse_attention_qkv true \
    --fuse_attention_ffn true \
    --enable_linear_fused_grad_add true \
    --data_cache "./data_cache" 2>&1 | tee log_8gpus_autoconfig.txt
