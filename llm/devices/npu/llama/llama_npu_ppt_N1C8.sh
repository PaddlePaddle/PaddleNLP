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

max_steps=${1:-2000}

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
ps aux | grep run_pretrain.py | grep -v grep | awk '{print $2}' | xargs kill -9

python -u  -m paddle.distributed.launch \
    --log_dir "./ppt_bf16_llama_N1C8" \
    ../run_pretrain.py \
    --model_name_or_path "meta-llama/Llama-2-13b" \
    --tokenizer_name_or_path "meta-llama/Llama-2-13b" \
    --input_dir "./pre-data" \
    --output_dir "./output" \
    --split 949,50,1 \
    --max_seq_length 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --per_device_eval_batch_size 1 \
    --use_flash_attention 1 \
    --use_fused_rms_norm 1 \
    --virtual_pp_degree 1 \
    --learning_rate 0.00001 \
    --min_learning_rate 0.000001 \
    --max_steps ${max_steps} \
    --save_steps 2000 \
    --seed 100 \
    --warmup_steps 20 \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --dataloader_num_workers 1 \
    --eval_steps 1001 \
    --tensor_parallel_degree 4 \
    --disable_tqdm true \
    --continue_training 0 \
    --do_train \
    --device "npu" \
    --enable_linear_fused_grad_add false \
    --fuse_attention_qkv true \
    --fuse_attention_ffn true \
    --use_fused_rope true \
    --recompute_use_reentrant true \
    --data_cache "./data_cache" \
    --bf16 \
    --fp16_opt_level "O2" \
    --amp_master_grad \
    --load_sharded_model true \
    --save_sharded_model true \
    --pipeline_parallel_degree 1 \
    --ignore_data_skip 0 \
    --force_reshard_pp true \
    --unified_checkpoint \
    --tensor_parallel_config "enable_mp_async_allreduce enable_mp_skip_c_identity" \
    --sequence_parallel 1 \
    --pipeline_parallel_config "disable_partial_send_recv" \
    --sharding "stage1" \
    --sharding_parallel_degree 2
