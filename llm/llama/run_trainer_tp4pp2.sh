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

set -x
unset CUDA_VISIBLE_DEVICES

export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export NVIDIA_TF32_OVERRIDE=0
export NCCL_ALGO=Tree
export Flags_mp_aysnc_allreduce=1
export Flags_skip_mp_c_identity=1
export FLAGS_shard_norm_align_dp=0
export FLAGS_shard_use_reduce=1


task_name="llama_hybrid"
# rm -rf /ssd2/zhonghui03/output/$task_name/
rm -rf "/ssd2/zhonghui03/output/$task_name""_log"


PYTHONPATH=../../:$PYTHONPATH  \
python -u  -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir "/ssd2/zhonghui03/output/$task_name""_log" \
    run_pretrain.py \
    --model_type "llama" \
    --model_name_or_path "facebook/llama-7b" \
    --tokenizer_name_or_path "facebook/llama-7b" \
    --input_dir "./data" \
    --output_dir "/ssd2/zhonghui03/output/$task_name" \
    --split 949,50,1 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 4 \
    --use_flash_attention 0 \
    --use_fused_rms_norm 0 \
    --fp16  \
    --fp16_opt_level "O2"  \
    --scale_loss 512 \
    --tensor_parallel_degree 1 \
    --pipeline_parallel_degree 4 \
    --virtual_pp_degree 1 \
    --sequence_parallel 0 \
    --learning_rate 0.00001 \
    --min_learning_rate 0.000001 \
    --max_steps 10000 \
    --save_steps 20 \
    --save_total_limit 2 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --dataloader_num_workers 1 \
    --eval_steps 1000 \
    --report_to "visualdl" \
    --sharding "" \
    --disable_tqdm true \
    --continue_training 0 \
    --recompute 1 \
    --recompute_granularity full \
    --unified_checkpoint 1 \
    --do_train \
    --do_eval \
    --device "gpu" \
    --distributed_dataloader 0
    # --pipeline_parallel_config "disable_partial_send_recv"  # if set sequence_parallel True, please note off this line.
    # reompute settings:
    # --no_recompute_layers 0 1 2 3 4 5 6 7 8 9 10 ... int int
    # --pp_recompute_interval 0 # A value of 0 indicates no recomputation.
