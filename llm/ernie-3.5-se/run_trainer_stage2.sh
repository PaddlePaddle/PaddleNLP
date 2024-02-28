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

task_name="ernie35_hybrid"
# rm -rf output/$task_name/
# rm -rf "output/$task_name""_log"

data_dir="./data"

PYTHONPATH=../../:$PYTHONPATH  \
python -u  -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir "output/$task_name""_log" \
    run_pretrain.py \
    --model_type "ernie" \
    --model_name_or_path "baidu/ernie-3.5-se-3b" \
    --tokenizer_name_or_path "ernie-tokenizer" \
    --input_dir "${data_dir}" \
    --output_dir "output/$task_name" \
    --split 949,50,1 \
    --max_seq_length 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 2 \
    --use_flash_attention 1 \
    --use_fused_ln 1 \
    --amp_master_grad 0 \
    --bf16 1 \
    --fp16_opt_level "O2"  \
    --scale_loss 512 \
    --tensor_parallel_degree 1 \
    --pipeline_parallel_degree 1 \
    --virtual_pp_degree 1 \
    --learning_rate 0.0003 \
    --min_learning_rate 0.00003 \
    --lr_scheduler_type "cosine" \
    --max_steps 300000 \
    --save_steps 200 \
    --adam_beta2 0.95 \
    --weight_decay 0.1 \
    --warmup_steps 2000 \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --dataloader_num_workers 0 \
    --eval_steps 50 \
    --report_to "visualdl" \
    --sharding "stage2" \
    --sharding_parallel_degree 8 \
    --disable_tqdm true \
    --continue_training 0 \
    --recompute 0 \
    --do_train \
    --do_eval \
    --save_total_limit 5 \
    --device "gpu"
