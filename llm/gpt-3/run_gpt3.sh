#!/bin/bash

set -x
#unset CUDA_VISIBLE_DEVICES
  
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
export PADDLE_NNODES=1

#which python
#pip install tool_helpers

#cd /work/PaddleNLP/model_zoo/gpt-3/external_ops
#python setup.py install

#export FLAGS_use_stride_kernel=0
#export FLAGS_cudnn_deterministic=1
#export FLAGS_embedding_deterministic=1
#export GLOG_vmodule=dygraph_functions=6,utils=6
#export GLOG_vmodule=layer=4
#export GLOG_vmodule=process_group_nccl=3

WORLD_SIZE=8
GBS=32
MBS=1
MP=2
SP=1  # 0 or 1
PP=4
VPP=1
SD=$(($WORLD_SIZE / ($MP * $PP)))
ACC_STEPS=$(($GBS / ($SD * $MBS)))
SEQLEN=2048

MODEL_TYPE="gpt3-13B-en"
#cp gpt3-13B-en.json /root/.paddlenlp/models/gpt3-13B-en/config.json

OUTPUT_FILENAME=paddle_${MODEL_TYPE}.gbs${GBS}_mp${MP}pp${PP}sd${SD}_vpp${VPP}_mbs${MBS}_acc${ACC_STEPS}.20240311

recompute_args="--recompute 1 \
                --recompute_use_reentrant true \
                --recompute_granularity full \
                --pp_recompute_interval 1"

#autoconfig_args="--auto_tuner_json ./benchmark/llama13b_pretrain_autoconfig.json"

if [ "$autoconfig_args" = "" ]; then
  #if [ "$MP" != "1" ]; then
  #  export CUDA_DEVICE_MAX_CONNECTIONS=1
  #fi
  if [ "$SP" = "1" ]; then
    extra_pp_config="disable_partial_send_recv"
  fi
fi

rm -rf log_${MODEL_TYPE}
rm -rf output

#export PATH=/opt/nvidia/nsight-systems/2023.4.1/bin:$PATH

#nsys_args="nsys profile --stats true -w true -t cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi -x true --force-overwrite true -o ${OUTPUT_FILENAME}"

${nsys_args} python3 -u -m paddle.distributed.launch \
        --gpus "${CUDA_VISIBLE_DEVICES}" ${autoconfig_args} \
        --log_dir log_${MODEL_TYPE} \
        run_pretrain.py \
        --model_name_or_path "${MODEL_TYPE}.json" \
        --tokenizer_name_or_path "${MODEL_TYPE}" \
        --input_dir "data" \
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
        --max_steps 30 \
        --eval_steps 1000 \
        --save_steps 5000 \
        --logging_steps 1 \
        ${recompute_args} \
        --dataloader_num_workers 1 \
        --use_flash_attention true \
        --use_fused_rms_norm true \
        --use_fast_layer_norm true \
		--use_fused_linear false	\
		--use_fused_dropout_add false \
		--fuse_attention_qkv true \
        --use_fused_rope true \
        --enable_linear_fused_grad_add false \
        --sharding "stage1" \
        --sharding_parallel_config "enable_stage1_tensor_fusion enable_stage1_overlap" \
        --tensor_parallel_config "enable_mp_async_allreduce enable_mp_skip_c_identity enable_mp_fused_linear_param_grad_add" \
        --pipeline_parallel_config "enable_sharding_comm_overlap ${extra_pp_config}" \
        --disable_tqdm true \
        --continue_training 0 \
        --device "gpu" 2>&1 | tee log_${OUTPUT_FILENAME}.txt

        #--sharding_parallel_config "split_param" \
