echo "lock_seed_flag : $lock_seed_flag"
if [[ ${lock_seed_flag} =~ "open_lock_seed" ]];then
    export npu_deterministic=true
    export ACL_OP_DETERMINISTIC=true
    export ACL_OPT_DETERMINISTIC=true
    export HCCL_DETERMINISTIC=true
    echo "npu_deterministic : $npu_deterministic   ACL_OP_DETERMINISTIC : $ACL_OP_DETERMINISTIC   ACL_OPT_DETERMINISTIC : $ACL_OPT_DETERMINISTIC   HCCL_DETERMINISTIC : $HCCL_DETERMINISTIC"
fi

echo "bf16_flag : $bf16_flag"
if [[ ${bf16_flag} =~ "open_bf16" ]];then
    export train_add_params=" --bf16 true "
else
    export train_add_params=" --fp16 true "
fi

echo "continue_training : $continue_training"
if [[ ${continue_training} =~ "close_continue_training" ]];then
    export train_add_params="$train_add_params --continue_training 0 "
elif [[ ${continue_training} =~ "no_continue_training" ]];then
    echo "dont add this params"
else
    export train_add_params="$train_add_params --continue_training 1 "
fi

if [[ -f "/usr/local/Ascend/atb/set_env.sh" ]]; then
    source /usr/local/Ascend/atb/set_env.sh
fi
if [[ -f "/usr/local/Ascend/mindie/latest/mindie-rt/mindie-atb/atb/set_env.sh" ]]; then
    source /usr/local/Ascend/mindie/latest/mindie-rt/mindie-atb/atb/set_env.sh
fi
if [[ -f "set_env.sh" ]];then #240517 添加
    source /usr/local/Ascend/mindie/set_env.sh
fi
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export nnodes=$PADDLE_TRAINERS_NUM
export rank=$PADDLE_TRAINER_ID
export START_NODE=0

export MASTER_IP=`echo ${PADDLE_TRAINERS} | awk -F, '{print $1}'`

if [[ -d "/opt/py39/lib/python3.9/site-packages/paddle_custom_device" ]]; then
     export CUSTOM_DEVICE_ROOT=/opt/py39/lib/python3.9/site-packages/paddle_custom_device
 fi

export FLAGS_NPU_MC2=1
export FLAGS_set_to_1d=False
export NVIDIA_TF32_OVERRIDE=0
export PADDLE_XCCL_BACKEND=npu
export FLAGS_use_stride_kernel=0

export MULTI_STREAM_MEMORY_REUSE=1
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
export FLAGS_eager_communication_connection=1
export FLAGS_allocator_strategy=naive_best_fit

ps aux | grep "train.py" | grep -v grep | awk '{print $2}' | xargs -r kill -9
ps aux | grep "run_pretrain.py" | grep -v grep | awk '{print $2}' | xargs -r kill -9
ps aux | grep "finetune_generation.py" | grep -v grep | awk '{print $2}' | xargs -r kill -9
ps aux | grep "dpo_train.py" | grep -v grep | awk '{print $2}' | xargs -r kill -9