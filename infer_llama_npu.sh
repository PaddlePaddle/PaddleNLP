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

