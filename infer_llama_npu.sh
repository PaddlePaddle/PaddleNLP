#! /bin/bash

use_device="0,1,2,3,4,5,6,7"
visible_device_="8,9,10,11,12,13,14,15"
# use_device="0"
export ASCEND_RT_VISIBLE_DEVICES=$visible_device_
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

#EVENT消减
export FLAGS_use_stream_safe_cuda_allocator=0

# 使能Lccl
export LCCL_ENABLE_FALLBACK=1
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
# export ATB_PROFILING_ENABLE=1

export FLAGS_eager_delete_tensor_gb=-1.0 
# export FLAGS_new_executor_serial_run=1
export FLAGS_allocator_strategy="auto_growth"

# model_dir="/home/data/acltransformer_testdata/weights/inference-dy7b-mp8/"
model_dir="/home/data/dataset/llama-7b/topp/inference-mp8-topp"
# model_dir="/home/data/dataset/llama-7b/inference-7b-dy-mp8-be61dce"
# 切换模型注意修改custom op
# model_dir="/home/data/dataset/llama-65b/inference-65b-dy-be61dce-topp"
#  model_dir="/home/data/dataset/llama-65b/inference-65b-dy-be61dce/"

log_dir=mp8
rm -rf $log_dir

python -m paddle.distributed.launch --log_dir $log_dir --devices $use_device python llm/predictor.py --model_name_or_path $model_dir --batch_size 8 --dtype "float16" --mode "static" --device "npu" --benchmark --inference_model 


# declare -A map
# # 需要通过lspci -vvv -s <ID>,确认每个卡numa亲和性
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
# BATCH_NUM=8

# if test -d "$model_dir";
# then
#     echo "Weight directory exists, runing......"
#     for((RANK_ID=$RANK_ID_START;RANK_ID<$WORLD_SIZE;RANK_ID++));
#     do
#     bind=${map["$RANK_ID"]}
#     echo "Device ID: $RANK_ID, bind to NUMA node: $bind"
#     numactl --cpunodebind=$bind --membind $bind python -m paddle.distributed.launch --master 127.0.0.1:49123 --rank $RANK_ID --devices $RANK_ID --nnodes $WORLD_SIZE python3 llm/predictor.py --model_name_or_path $model_dir --batch_size $BATCH_NUM --dtype "float16" --mode "static" --inference_model 1 &> $RANK_ID.log &
# done
# # tail -f 0.log


