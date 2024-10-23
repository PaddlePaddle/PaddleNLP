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

max_steps=${1:-800}

MODEL_PATH=""
DATASET_NAME_OR_PATH=""

set -x
export MC2=1
export GLOG_v=0
export FLAGS_npu_storage_format=1
export HCCL_INTRA_PCIE_EHABLE=0
export HCCL_INTRA_ROCE_ENABLE=1
export FLAGS_allocator_strategy=naive_best_fit

export FLAGS_NPU_MC2=1
export MC2_Recompute=1
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS

export FLAGS_use_stride_kernel=0
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
export MULTI_STREAM_MEMORY_REUSE=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=../../../:$PYTHONPATH
ps aux | grep run_pretrain.py | grep -v grep | awk '{print $2}' | xargs kill -9

python3 -u  -m paddle.distributed.launch \
    --log_dir "./log_ppt_baichuan2_13b" \
    ../../run_pretrain.py \
    --model_name_or_path "${MODEL_PATH}" \
    --tokenizer_name_or_path "${DATASET_NAME_OR_PATH}" \
    --input_dir "./pre-data" \
    --output_dir "./output" \
    --split 949,50,1 \
    --max_seq_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 1 \
    --use_flash_attention false \
    --use_fused_rms_norm 1 \
    --virtual_pp_degree 1 \
    --learning_rate 0.00001 \
    --min_learning_rate 0.000001 \
    --max_steps ${max_steps} \
    --decay_steps 2000 \
    --save_steps 2000 \
    --seed 100 \
    --weight_decay 0.01 \
    --warmup_steps 20 \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --dataloader_num_workers 1 \
    --eval_steps 1001 \
    --tensor_parallel_degree 8 \
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
    --tensor_parallel_config "enable_mp_async_allreduce enable_mp_skip_c_identity" \
    --sequence_parallel 1 \
    --pipeline_parallel_config "disable_partial_send_recv" \
    --sharding "stage1" \
    --sharding_parallel_degree 1