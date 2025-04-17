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

max_steps=${1:-100}

export GLOG_v=0
export FLAGS_npu_storage_format=1
export FLAGS_use_stride_kernel=0
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
export MULTI_STREAM_MEMORY_REUSE=1
export HCCL_INTRA_PCIE_ENABLE=0
export HCCL_INTRA_ROCE_ENABLE=1
export FLAGS_allocator_strategy=naive_best_fit
export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export FLAGS_NPU_MC2=1
export MC2_Recompute=1

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=../../../../:$PYTHONPATH
ps aux | grep run_finetune.py | grep -v grep | awk '{print $2}' | xargs kill -9

python -u  -m paddle.distributed.launch \
    --devices "0,1,2,3,4,5,6,7" \
    --log_dir "./lora_bf16_llama_N1C8" \
    ../../../run_finetune.py \
    --device "npu" \
    --model_name_or_path "meta-llama/Llama-2-13b-chat" \
    --dataset_name_or_path "data/" \
    --output_dir "./output/lora_bf16_llama_N1C8" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 64 \
    --per_device_eval_batch_size 1 \
    --eval_accumulation_steps 1 \
    --max_steps ${max_steps} \
    --learning_rate 3e-06 \
    --warmup_steps 20 \
    --save_steps 1000 \
    --logging_steps 1 \
    --evaluation_strategy "epoch" \
    --src_length 1024 \
    --tensor_parallel_output true \
    --max_length 4096 \
    --bf16 true \
    --fp16_opt_level "O2" \
    --do_train true \
    --disable_tqdm true \
    --eval_with_do_generation false \
    --metric_for_best_model "accuracy" \
    --recompute false \
    --tensor_parallel_degree 8 \
    --pipeline_parallel_degree 1 \
    --sharding_parallel_degree 1 \
    --zero_padding 0 \
    --sequence_parallel 1 \
    --amp_master_grad true \
    --fuse_attention_qkv true \
    --fuse_attention_ffn true \
    --use_flash_attention 1 \
    --use_fused_rope 1 \
    --use_fused_rms_norm 1 \
    --lora true \
    --lora_rank 32 \
    --pad_to_multiple_of 4096
