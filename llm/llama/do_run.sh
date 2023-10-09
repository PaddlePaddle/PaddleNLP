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

export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export FLAGS_flash_attn_version=v1
export USE_FAST_LN=0

export FLAGS_shard_norm_align_dp=1
export FLAGS_shard_use_reduce=0
export FLAGS_enable_sep_shard=0

mode="tp"
mode="sep"
if [[ $# > 0 ]]; then
  mode=$1
fi

# max_seq_length=16384
max_seq_length=65536
if [[ $# > 1 ]]; then
  max_seq_length=$2
fi

gpus="0,1,2,3,4,5,6,7"
if [[ $# > 2 ]]; then
  gpus=$3
fi

gpu_num=8
if [[ $# > 3 ]]; then
  gpu_num=$4
fi

nnodes=1
if [[ $# > 3 ]]; then
  nnodes=$5
fi

echo "mode:$mode, max_seq_length:$max_seq_length, gpus:$gpus, gpu_num:$gpu_num, nnodes:$nnodes"

rank=${PADDLE_TRAINER_ID-0}
if [[ $nnodes -gt 1 ]]; then
  master=`cat /root/paddlejob/workspace/hostfile | head -n 1 | awk '{print $1}'`
else
  master=127.0.0.1
fi
port=36677

max_steps=10
log_dir=${mode}_log_seq_${max_seq_length}_gpus_${gpu_num}_nodes_${nnodes}
echo "log_dir:${log_dir}"
# log_dir=tp_log_tmp

export PYTHONPATH=../../:$PYTHONPATH
if [[ $mode == "tp" ]]; then
rm -rf dp_input_data/*
# rm -rf tp_log
# nsys profile --stats true -w true -t cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi -x true --force-overwrite true -o test_paddle \
python -u  -m paddle.distributed.launch \
    --master $master:$port \
    --nnodes $nnodes \
    --rank $rank \
    --gpus $gpus \
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
    --gradient_accumulation_steps 8 \
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
    --rope_fusion_level "core" \
    --enable_linear_fused_grad_add false \
    --fuse_attention_qkv true \
    --fuse_attention_ffn true \
    --tensor_parallel_config "enable_mp_async_allreduce enable_mp_skip_c_identity enable_mp_fused_linear_param_grad_add" \
    --recompute_use_reentrant true \
    --data_cache "./data_cache" \
    --pipeline_parallel_degree 1 \
    --sep_parallel_degree 1 \
    --tensor_parallel_degree $gpu_num \
    --sequence_parallel true \
    --skip_profile_timer true \
    --amp_master_grad \
    # --sharding "stage1" \
    # --sharding_parallel_degree 8 \

elif [[ $mode == "sep" ]]; then
# rm -rf sep_log
# nsys profile --stats true -w true -t cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi -x true --force-overwrite true -o test_paddle \
python -u  -m paddle.distributed.launch \
    --master $master:$port \
    --nnodes $nnodes \
    --rank $rank \
    --gpus $gpus \
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
    --gradient_accumulation_steps 8 \
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
    --rope_fusion_level "core" \
    --enable_linear_fused_grad_add false \
    --fuse_attention_qkv true \
    --fuse_attention_ffn true \
    --tensor_parallel_config "enable_mp_async_allreduce enable_mp_skip_c_identity enable_mp_fused_linear_param_grad_add" \
    --recompute_use_reentrant true \
    --data_cache "./data_cache" \
    --pipeline_parallel_degree 1 \
    --sep_parallel_degree $gpu_num \
    --tensor_parallel_degree 1 \
    --sequence_parallel false \
    --skip_profile_timer true \
    --amp_master_grad \
    # --sharding "stage1" \
    # --sharding_parallel_degree 8 \

fi

