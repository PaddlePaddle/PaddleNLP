#!/bin/bash

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

set -ex

PNLP_PATH="/workspace/PaddleNLP"
OUTPUT_BASE=${OUTPUT_BASE:="/workspace/outputs"}
DATA_PATH=/dataset
SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:=1}

export PYTHONPATH="${PNLP_PATH}:${PYTHONPATH}"
export NVTE_FUSED_ATTN=1
#export NVTE_ASYNC_AMAX_REDUCTION=1  # enable async allreduce
#export CUDA_DEVICE_MAX_CONNECTIONS=1

per_device_batch_size=${1:-6}
fsdp=${2:-2}
tp=${3:-4}
pp=${4:-1}
sharding_stage=${5:-"stage2"}
backend=${6:-"none"}
precision=${7:-"bf16"}
recompute=${8:-"full"}
resume_step=${9:-"none"}
init_weigth=${10:-"none"}
nsys_profile=${11:-"false"}
vp=${12:-1}
ga=${13:-1}
sp=${14:-"false"}
model_name=${15:-"gpt3-1.3B-en"}
tokenizer_name=${16:-"gpt3-1.3B-en"}

dp=`expr $((8*SLURM_JOB_NUM_NODES)) / ${tp} / ${fsdp}`

log_dir=${EXP_NAME:="N${SLURM_JOB_NUM_NODES}_DP${dp}_TP${tp}_PP${pp}_VP${vp}_GA${ga}_SP${sp}_FSDP${fsdp}_MBS${per_device_batch_size}_${backend}_${precision}_${recompute}_${model_name}"}
#rm -rf $log_dir
output_dir="${OUTPUT_BASE}/$log_dir"

IP_STR=${IP_STR:='127.0.0.1'}

if [ "${backend}" == "none" ]; then
   readonly backend_flag=""
elif [ "${backend}" == "te" ]; then
   readonly backend_flag=" --transformer_engine_backend transformer_engine "
elif [ "${backend}" == "pd" ]; then
   readonly backend_flag=" --transformer_engine_backend paddle "
else
   echo "Error! backend=${backend} not supported!"
   return -1
fi

if [ "${sharding_stage}" == "stage1" ]; then
   readonly stage_flag=" --sharding stage1 "
elif [ "${sharding_stage}" == "stage2" ]; then
   readonly stage_flag=" --sharding stage2 "
elif [ "${sharding_stage}" == "stage3" ]; then
   readonly stage_flag=" --sharding stage3 "
else
   echo "Error! sharding stage=${sharding_stage} not supported!"
   return -1
fi

if [ "${precision}" == "bf16" ]; then
   readonly precision_flag=""
elif [ "${precision}" == "fp8" ]; then
   # fp8 precision is only supported by te backend
   if [ "${backend}" != "te" ]; then
       echo "Error! fp8 precision is only supported by te backend!"
       return -1
   fi
   readonly precision_flag=" --use_fp8 "
else
   echo "Error! precision=${precision} not supported!"
   return -1
fi

if [ "${recompute}" == "none" ]; then
   readonly recompute_flag=""
elif [ "${recompute}" == "core_attn" ]; then
   readonly recompute_flag=" --recompute 1 --recompute_granularity core_attn "
elif [ "${recompute}" == "full" ]; then
   readonly recompute_flag=" --recompute 1 --recompute_granularity full "
else
   echo "Error! recompute=${recompute} not supported!"
   return -1
fi

if [ "${resume_step}" == "none" ]; then
    readonly resume_flag=""
elif [ "${backend}" == "auto" ]; then
    readonly resume_flag=""
else # ckpt_step is a number
    ckpt_path="$output_dir/checkpoint-${resume_step}"
    # assert ckpt_path exists
    if [ ! -d "$ckpt_path" ]; then
        echo "Error! ckpt_path=${ckpt_path} not exists!"
        return -1
    fi
    readonly resume_flag=" --resume_from_checkpoint $ckpt_path "
fi

if [ "${init_weigth}" == "none" ]; then
   readonly init_weigth_flag=""
else
   readonly init_weigth_flag=" --te_init_weight_path ${init_weigth} "
fi

if [ "${nsys_profile}" == "false" ]; then
   readonly nsys_cmd=""
   export ENABLE_PROFILE=0
elif [ "${nsys_profile}" == "true" ]; then
   export ENABLE_PROFILE=1
   export PROFILE_START_STEP=3
   export PROFILE_STOP_STEP=3
   export PROFILE_EMIT_NVTX=1
   readonly nsys_cmd="nsys profile -s none -c cudaProfilerApi -t cuda,nvtx --force-overwrite true --capture-range-end=stop -o ${output_dir}/profile/${log_dir} "
   #readonly nsys_cmd="nsys profile -s none -c cudaProfilerApi -t cuda,nvtx --force-overwrite true -o ${output_dir}/profile/${log_dir} "
   mkdir -p ${output_dir}/profile
else
   echo "Error! nsys_profile=${nsys_profile} not supported!"
   return -1
fi

sp_flag=""
if [ "${sp}" == "true" ]; then
   sp_flag="--sequence_parallel --pipeline_parallel_config=disable_partial_send_recv"
fi

${nsys_cmd} python -u  -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --ips="${IP_STR}" \
    --log_dir ${output_dir}/logs \
    ${PNLP_PATH}/llm/gpt-3/run_pretrain_with_te.py \
    --model_name_or_path ${model_name} \
    --tokenizer_name_or_path ${tokenizer_name} \
    --input_dir "$DATA_PATH" \
    --output_dir $output_dir/output \
    --split 949,50,1 \
    --max_seq_length 2048 \
    --per_device_train_batch_size $per_device_batch_size \
    --per_device_eval_batch_size $per_device_batch_size \
    --tensor_parallel_degree $tp \
    --sharding_parallel_degree $fsdp \
    --pipeline_parallel_degree $pp \
    --fuse_attention_qkv 1 \
    --use_flash_attention 0 \
    --bf16  \
    --fp16_opt_level "O2"  \
    --scale_loss 1024 \
    --learning_rate 0.00001 \
    --min_learning_rate 0.000005 \
    --max_steps 20000 \
    --save_steps 200000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --dataloader_num_workers 1 \
    --hidden_dropout_prob=0.1 \
    --attention_probs_dropout_prob=0.1 \
    --eval_steps 1000 \
    --report_to "visualdl" \
    --disable_tqdm true \
    --gradient_accumulation_steps ${ga} \
    --do_train \
    --continue_training 0 \
    --virtual_pp_degree ${vp} \
    $stage_flag \
    $resume_flag \
    $backend_flag \
    $precision_flag \
    $recompute_flag \
    $init_weigth_flag \
    $sp_flag \
    --device "gpu"
