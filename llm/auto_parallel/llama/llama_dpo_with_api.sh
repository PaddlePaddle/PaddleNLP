# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
export FLAGS_auto_parallel_aligm_mode=0

task_name="llama_dpo_auto_with_api"
rm -rf output/$task_name/
rm -rf "log/$task_name""_log"

export SOT_LOG_LEVEL=4
export PYTHONPATH=../../../:$PYTHONPATH

python -u  -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir  "log/$task_name""_log" \
    ../run_dpo_auto.py \
    --model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
    --train_dataset_path "../../data/train.jsonl" \
    --dev_dataset_path "../../data/dev.jsonl" \
    --output_dir "./checkpoints/dpo_ckpts" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 1 \
    --max_steps 10 \
    --learning_rate 1e-06 \
    --warmup_steps 10 \
    --logging_steps 1 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --eval_steps 100 \
    --save_steps 500 \
    --max_seq_len 4096 \
    --max_prompt_len 2048 \
    --bf16 true \
    --fp16_opt_level "O2" \
    --do_train true \
    --do_eval true \
    --disable_tqdm true \
    --load_best_model_at_end true \
    --tensor_parallel_degree 8 \
    --sharding "stage1" \
    --use_flash_attention true \
    --flash_mask true \
    --recompute false \
    --recompute_granularity "full" \
    --beta 0.1 \
    --benchmark false \
    --loss_type "sigmoid" \
    --label_smoothing 0.0 \
    --unified_checkpoint true \
    --autotuner_benchmark false \
    --lazy false \
    --max_grad_norm 0.0 \
    --seed 42 \
    --to_static true \
    --enable_auto_parallel true \
    --use_intermediate_api true \
