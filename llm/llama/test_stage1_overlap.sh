#!/bin/bash

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

set -x
#unset CUDA_VISIBLE_DEVICES
  
WORK_ROOT=/root/paddlejob/workspace/yinwei
export PYTHONPATH=${WORK_ROOT}/PaddleNLP:$PYTHONPATH

export FLAGS_selected_gpus="0,1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
export PADDLE_NNODES=1

export FLAGS_cudnn_deterministic=1
export FLAGS_embedding_deterministic=1

#which python
#pip install tool_helpers

#cd /work/PaddleNLP/model_zoo/gpt-3/external_ops
#python setup.py install

#export FLAGS_use_stride_kernel=0
#export FLAGS_cudnn_deterministic=1
#export FLAGS_embedding_deterministic=1
#export GLOG_vmodule=dygraph_functions=6,utils=6
#export GLOG_vmodule=layer=4

WORLD_SIZE=8
GBS=8
MBS=1
MP=1
SP=0 # 0 or 1
PP=1
DP=1
VPP=1
SD=$(($WORLD_SIZE / ($MP * $PP * $DP)))
ACC_STEPS=$(($GBS / ($SD * $MBS)))

COMPANY_NAME="facebook"
# COMPANY_NAME="meta-llama"
# COMPANY_NAME="__internal_testing__"

MODEL_TYPE="llama-7b"
# MODEL_TYPE="Llama-2-13b"
# MODEL_TYPE="tiny-random-llama"

# facebook/llama-7b

if [ "${COMPANY_NAME}" = "facebook" ]; then
  SEQLEN=2048
elif [ "${COMPANY_NAME}" = "meta-llama" ]; then
  SEQLEN=4096
elif [ "${COMPANY_NAME}" = "__internal_testing__" ]; then
  SEQLEN=2048
fi

recompute=1
if [ "${recompute}" = "1" ]; then
  # no_recompute_layers="--no_recompute_layers 24:40"
  recompute_granularity="full"
  recompute_args="--recompute 1 --recompute_granularity ${recompute_granularity} ${no_recompute_layers} --pp_recompute_interval 1 --recompute_use_reentrant true"
fi

#autoconfig_args="--auto_tuner_json ./benchmark/llama13b_pretrain_autoconfig.json"

Overlap=0
if [ "$autoconfig_args" = "" ]; then
  if [ "$MP" != "1" ]; then
    export CUDA_DEVICE_MAX_CONNECTIONS=1
  fi
  if [ "$SP" = "1" ]; then
    extra_pp_config="disable_partial_send_recv"
  fi
  if [ "$Overlap" = "1" ]; then
    OUTPUT_FILENAME=paddle_${MODEL_TYPE}.gbs${GBS}_mp${MP}pp${PP}sd${SD}_vpp${VPP}_mbs${MBS}_acc${ACC_STEPS}.20240319_stage1
  else
    OUTPUT_FILENAME=paddle_${MODEL_TYPE}.gbs${GBS}_mp${MP}pp${PP}sd${SD}_vpp${VPP}_mbs${MBS}_acc${ACC_STEPS}.20240312_stage1
  fi
else
  OUTPUT_FILENAME=paddle_${MODEL_TYPE}_gbs${GBS}_autoconfig.20231117
fi

rm -rf log_${MODEL_TYPE}
rm -rf output

# nsys_args="nsys profile --stats true -w true -t cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi -x true --force-overwrite true -o ${OUTPUT_FILENAME}"



OUTPUT_FILENAME="run_llama2_stage2-sharding8-no-overlap"
# OUTPUT_FILENAME="run_llama2_stage2-sharding8-overlap"
nsys_args="nsys profile --stats true -w true -t cuda,nvtx,cudnn,cublas -x true --force-overwrite true -o ${OUTPUT_FILENAME}"
${nsys_args} python -u -m paddle.distributed.launch \
        --gpus "0,1,2,3,4,5,6,7" ${autoconfig_args} \
        --log_dir log_${MODEL_TYPE} \
        run_pretrain.py \
        --model_name_or_path "${COMPANY_NAME}/${MODEL_TYPE}" \
        --tokenizer_name_or_path "${COMPANY_NAME}/${MODEL_TYPE}" \
        --input_dir "./data" \
        --output_dir "output" \
        --split 949,50,1 \
        --max_seq_length ${SEQLEN} \
        --per_device_train_batch_size ${MBS} \
        --gradient_accumulation_steps ${ACC_STEPS} \
        --per_device_eval_batch_size 4 \
        --bf16 \
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
        --max_steps 5 \
        --eval_steps 1000 \
        --save_steps 5000 \
        --logging_steps 2 \
        ${recompute_args} \
        --dataloader_num_workers 1 \
        --use_flash_attention true \
        --use_fused_rms_norm true \
        --fuse_attention_qkv true \
        --fuse_attention_ffn true \
        --use_fused_rope true \
        --enable_linear_fused_grad_add true \
        --sharding "stage2" \
        --sharding_degree ${SD} \
        --disable_tqdm true \
        --continue_training 0 \
        --device "gpu" 2>&1 | tee log_${OUTPUT_FILENAME}.txt


#         --sharding_parallel_config "enable_stage2_overlap" \