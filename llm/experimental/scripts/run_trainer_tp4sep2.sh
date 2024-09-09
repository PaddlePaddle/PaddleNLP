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

unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT

# export FLAGS_embedding_deterministic=1
# export FLAGS_cudnn_deterministic=1
# export FLAGS_flash_attn_version=v1
# export USE_FAST_LN=0


max_seq_length=16384

master=127.0.0.1
port=36677

max_steps=3
log_dir=seq_${max_seq_length}_log
echo "log_dir:${log_dir}"
rm -rf $log_dir

export PYTHONPATH=../../:$PYTHONPATH
python -u  -m paddle.distributed.launch \
    --master $master:$port \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir "./$log_dir" \
    run_pretrain.py \
    --model_type "llama" \
    --model_name_or_path "facebook/llama-7b" \
    --tokenizer_name_or_path "facebook/llama-7b" \
    --input_dir "./data" \
    --output_dir "./output" \
    --split 949,50,1 \
    --max_seq_length $max_seq_length \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 4 \
    --bf16  \
    --fp16_opt_level "O2"  \
    --use_flash_attention 1 \
    --use_fused_rms_norm 1 \
    --virtual_pp_degree 1 \
    --pp_recompute_interval 1 \
    --learning_rate 0.00001 \
    --min_learning_rate 0.000001 \
    --max_steps $max_steps \
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
    --use_fused_rope true \
    --enable_linear_fused_grad_add false \
    --fuse_attention_qkv true \
    --fuse_attention_ffn true \
    --tensor_parallel_config "enable_mp_async_allreduce enable_mp_skip_c_identity enable_mp_fused_linear_param_grad_add" \
    --recompute_use_reentrant true \
    --data_cache "./data_cache" \
    --pipeline_parallel_degree 1 \
    --sep_parallel_degree 2 \
    --tensor_parallel_degree 4 \
    --sequence_parallel false \
    --skip_profile_timer true \
    --amp_master_grad \

