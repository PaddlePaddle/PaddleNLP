#! /bin/bash

use_device="0,1,2,3,4,5,6,7"
visible_device_="8,9,10,11,12,13,14,15"
# use_device="0"
export ASCEND_RT_VISIBLE_DEVICES=$visible_device_

BATCH_NUM=8
SRC_LEN=3072
MAX_LEN=4096
# 加速库日志
# export ATB_LOG_TO_FILE=1
# export ATB_LOG_TO_STDOUT=1
# export ATB_LOG_LEVEL=INFO
# CANN日志
# export ASCEND_SLOG_PRINT_TO_STDOUT=1
# export ASCEND_GLOBAL_LOG_LEVEL=3
# 调试打开
# export FLAGS_npu_blocking_run=true
# export GLOG_v=4

# 落Tensor
# export ATB_SAVE_TENSOR=1
# export ATB_SAVE_TENSOR_END=0
# export ATB_SAVE_TENSOR_START=0
# export ATB_SAVE_TENSOR_RUNNER="SelfAttentionFusionOpsRunner"

#EVENT消减
export FLAGS_use_stream_safe_cuda_allocator=0

# 使能Lccl
# export LCCL_ENABLE_FALLBACK=1
# export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
export HCCL_BUFFSIZE=120

# 内存算法
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1

# export ATB_PROFILING_ENABLE=1

export FLAGS_eager_delete_tensor_gb=-1.0 
if [ $BATCH_NUM -gt 20 ] && [ $MAX_LEN -gt 4000 ];
then
    export FLAGS_eager_delete_tensor_gb=0.5
fi
# export FLAGS_new_executor_serial_run=1
# export FLAGS_allocator_strategy="auto_growth"

# model_dir="/home/data/acltransformer_testdata/weights/inference-dy7b-mp8/"
# model_dir="/home/data/dataset/llama-7b/topp/inference-mp8-topp"
# model_dir="/home/data/dataset/llama-7b/inference-7b-dy-mp8-be61dce"
# 切换模型注意修改custom op
# model_dir="/home/data/dataset/llama-65b/inference-65b-dy-be61dce-topp"
model_dir="/home/data/dataset/llama-65b/inference-65b-dy-be61dce/"

log_dir=mp8
rm -rf $log_dir

python -m paddle.distributed.launch --log_dir $log_dir --devices $use_device python llm/predictor.py --model_name_or_path $model_dir --batch_size $BATCH_NUM --src_length $SRC_LEN --max_length $MAX_LEN --dtype "float16" --mode "static" --device "npu" --inference_model 


# declare -A map
# # 1、通过npu-smi info，确认每个卡的Bus-Id
# # 2、通过lspci -vvv -s <Bus-Id>,确认每个卡numa node 亲和性
# map["0"]="0"
# map["1"]="0"
# map["2"]="0"
# map["3"]="0"
# map["4"]="1"
# map["5"]="1"
# map["6"]="1"
# map["7"]="1"

# RANK_ID_START=0
# WORLD_SIZE=8

# if test -d "$model_dir";
# then
#     echo "Weight directory exists, runing......"
#     for((RANK_ID=$RANK_ID_START;RANK_ID<$WORLD_SIZE;RANK_ID++));
#     do
#     bind=${map["$RANK_ID"]}
#     echo "Device ID: $RANK_ID, bind to NUMA node: $bind"
#     numactl --cpunodebind=$bind --membind $bind python -m paddle.distributed.launch --master 127.0.0.1:49123 --rank $RANK_ID --devices $RANK_ID --nnodes $WORLD_SIZE python3 llm/predictor.py --model_name_or_path $model_dir --batch_size $BATCH_NUM --src_length $SRC_LEN --max_length $MAX_LEN --dtype "float16" --mode "static" --device "npu" --benchmark --inference_model 1 &> $RANK_ID.log &
# done
# fi
# # tail -f 0.log


