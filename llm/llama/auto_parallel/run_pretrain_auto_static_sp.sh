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

# just for debug auto_parallel

set -x
unset CUDA_VISIBLE_DEVICES

task_name="llama_auto_dp2mp2pp2_vpp2_sp"
# rm -rf output/$task_name/  # ckpt is saved in 'output/''
rm -rf "output/$task_name""_log"

export PARALLEL_CROSS_ENTROPY=true
export FLAGS_call_stack_level=2
export PYTHONPATH=../../../:$PYTHONPATH
python -u -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir "output/$task_name""_log" \
    run_pretrain_auto_static.py \
    --model_type "llama" \
    --model_name_or_path "facebook/llama-7b" \
    --tokenizer_name_or_path "facebook/llama-7b" \
    --input_dir "./data" \
    --output_dir "output/$task_name" \
    --split 949,50,1 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --use_flash_attention 0 \
    --use_fused_rms_norm 0 \
    --fp16 0 \
    --fp16_opt_level "O2"  \
    --scale_loss 1024 \
    --tensor_parallel_degree 2 \
    --pipeline_parallel_degree 2 \
    --virtual_pp_degree 2 \
    --pipeline_schedule_mode "VPP" \
    --sharding_parallel_degree 1 \
    --sharding "stage2" \
    --learning_rate 0.0001 \
    --min_learning_rate 0.00001 \
    --max_steps 10 \
    --save_steps 5000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --dataloader_num_workers 1 \
    --eval_steps 1000 \
    --report_to "visualdl" \
    --disable_tqdm true \
    --continue_training 0 \
    --recompute 1 \
    --recompute_granularity full \
    --do_train \
    --do_eval \
    --device "gpu" \
    --data_impl "mmap" \
    --enable_auto_parallel 1 \
    --sequence_parallel true \

    # --resume_from_checkpoint "output/llama_auto_serial/checkpoint-2" \
