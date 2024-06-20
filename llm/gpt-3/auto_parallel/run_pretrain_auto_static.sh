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

export GLOG_v=0
export PYTHONPATH="../../../":$PYTHONPATH
export TRANSLATOR_DISABLE_NEW_ERROR=0
export TRANSLATOR_CODE_LEVEL=100
export FLAGS_call_stack_level=3

unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
export PADDLE_NNODES=1

task_name="gpt3_auto_static"
log_dir="log/$task_name"
rm -rf $log_dir
output_dir="output/$task_name"
input_dir="../data"

WORLD_SIZE=8
GBS=1 # default: 32, debug 4
MBS=1
MP=2
PP=4
VPP=1
SD=$(($WORLD_SIZE / ($MP * $PP)))  # 8 / (2*4) = 1
ACC_STEPS=$(($GBS / ($SD * $MBS))) # 32/(1*1) = 32
SEQLEN=4096

recompute_args="--recompute 1 \
                --recompute_use_reentrant true \
                --recompute_granularity full \
                --pp_recompute_interval 1"

use_flash_attention=0
use_fused_linear=1
enable_linear_fused_grad_add=1
fuse_attention_qkv=1
use_fused_dropout_add=1
use_fast_layer_norm=0
fuse_allreduce_split_to_reducescatter=0

SP=0  # 0 or 1
if [ "$MP" != "1" ]; then
  export CUDA_DEVICE_MAX_CONNECTIONS=1
fi
if [ "$SP" = "1" ]; then
  extra_pp_config="disable_partial_send_recv"
fi

# MODEL_TYPE="gpt3-13B-en"
MODEL_TYPE="gpt2-medium-en"

python -u -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir ${log_dir} \
    run_pretrain_auto.py \
    --model_name_or_path "${MODEL_TYPE}" \
    --tokenizer_name_or_path "${MODEL_TYPE}" \
    --input_dir ${input_dir} \
    --output_dir ${output_dir}  \
    --split 949,50,1 \
    --max_seq_length ${SEQLEN} \
    --per_device_train_batch_size ${MBS} \
    --gradient_accumulation_steps ${ACC_STEPS} \
    --per_device_eval_batch_size 4 \
    --bf16 0 \
    --fp16_opt_level "O2"  \
    --amp_master_grad true \
    --tensor_parallel_degree ${MP} \
    --pipeline_parallel_degree ${PP} \
    --virtual_pp_degree ${VPP} \
    --sequence_parallel ${SP} \
    --learning_rate 0.00001 \
    --min_learning_rate 0.000001 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --do_train \
    --max_steps 30 \
    --eval_steps 1000 \
    --save_steps 5000 \
    --logging_steps 1 \
    ${recompute_args} \
    --dataloader_num_workers 1 \
    --use_flash_attention ${use_flash_attention} \
    --fuse_attention_qkv ${fuse_attention_qkv} \
    --use_fast_layer_norm ${use_fast_layer_norm} \
    --fused_linear ${use_fused_linear} \
    --fused_linear_param_grad_add ${enable_linear_fused_grad_add} \
    --use_fused_dropout_add ${use_fused_dropout_add} \
    --fuse_allreduce_split_to_reducescatter ${fuse_allreduce_split_to_reducescatter} \
    --tensor_parallel_config "enable_mp_async_allreduce" \
    --pipeline_parallel_config "" \
    --sharding "" \
    --sharding_parallel_config "" \
    --disable_tqdm true \
    --continue_training 0 \
    --skip_memory_metrics 0 \
    --report_to "none" \
    --model_type "gpt" \
    --enable_auto_parallel 1 \
    --to_static 1 \
    --device "gpu"
    
