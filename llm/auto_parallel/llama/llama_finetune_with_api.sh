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

# just for debug

set -x

unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT

export NNODES=1
export PADDLE_TRAINERS_NUM=1

export GLOG_v=0

export FLAGS_cudnn_deterministic=0
export FLAGS_embedding_deterministic=0
# export FLAGS_max_inplace_grad_add=65536
export FLAGS_enable_auto_parallel_align_mode=0

task_name="llama_3.1_sft_auto"
rm -rf output/$task_name/
rm -rf "log/$task_name""_log"

export SOT_LOG_LEVEL=4
# export PYTHONPATH=../:$PYTHONPATH
export PYTHONPATH=../../../:$PYTHONPATH
#ulimit -c unlimited
to_static=true

python -u  -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir  "log/$task_name""_log" \
    ../run_finetune_auto.py \
    --model_name_or_path "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --dataset_name_or_path "./data" \
    --output_dir "output/$task_name/" \
    --enable_auto_parallel true \
    --lora false \
    --use_mora false \
    --model_type "llama_network" \
    --use_intermediate_api true \
    --to_static $to_static \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 8 \
    --eval_accumulation_steps 16 \
    --num_train_epochs 1 \
    --learning_rate 3e-05 \
    --max_steps 10 \
    --warmup_steps 30 \
    --logging_steps 1 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --src_length 1024 \
    --max_length 2048 \
    --bf16 true \
    --fp16_opt_level "O2" \
    --amp_master_grad true \
    --do_train true \
    --do_eval false \
    --disable_tqdm true \
    --load_best_model_at_end true \
    --eval_with_do_generation false \
    --metric_for_best_model "accuracy" \
    --recompute false \
    --save_total_limit 1 \
    --tensor_parallel_degree 2 \
    --pipeline_parallel_degree 2\
    --zero_padding false \
    --unified_checkpoint false \
    --flash_mask false \
    --use_flash_attention true \
    --fuse_attention_qkv true \
    --sharding "stage1" \
    --auto_parallel_resume_form_hybrid_parallel true \
    --num_hidden_layers 2 
