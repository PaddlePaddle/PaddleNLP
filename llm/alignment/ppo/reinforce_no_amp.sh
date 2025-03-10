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

unset PADDLE_TRAINERS_NUM
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
export PYTHONPATH="/root/paddlejob/workspace/gongenlei/PaddleNLP_20250309/llm/":$PYTHONPATH
current_time=37ppl_3600_steps_no_amp # for debug
rm -rf ${current_time}
mkdir -p ${current_time}
rm -rf core.*

remove_checkpoints=0
export FLAGS_set_to_1d=False
export NVIDIA_TF32_OVERRIDE=0
export FLAGS_dataloader_use_file_descriptor=False
export HF_DATASETS_DOWNLOAD_TIMEOUT=1
export FLAGS_enable_pir_api=0
# 如果需要精度对齐，请开启JINGDUDUIQI=True，（1）数据流固定，不shuffle，不开启balance batch
export JINGDUDUIQI=True # True or False
BOS_PROXY=http://x.x.x.x:x/
export http_proxy=$BOS_PROXY
export https_proxy=$BOS_PROXY
export no_proxy="127.0.0.1,0.0.0.0,localhost"
export WANDB_HTTPS_PROXY=$BOS_PROXY
export SAVE_OUTPUT=False

# 通用环境变量，避免随机性，精度对齐时开启
# export FLAGS_embedding_deterministic=1
# export FLAGS_cudnn_deterministic=1

# 生成，提升数值稳定性，精度对齐时开启
export FLAGS_gemm_use_half_precision_compute_type=False
export FLAGS_force_cublaslt_no_reduced_precision_reduction=True
export NCCL_ALGO=Tree
export WANDB_PROJECT="Qwen2.5-7B-Instruct-reinforce++_nonormalize"


export actor_model_name_or_path="/root/paddlejob/workspace/gongenlei/Qwen2.5-7B-Instruct-1M"
lr=5e-7
top_p=1.0
repetition_penalty=1.0
temperature=0.7
max_steps=3600

# log
file_flag="lr_${lr}_topp_${top_p}_max_steps_${max_steps}"
log_file="${current_time}/log_${file_flag}"
vdl="${current_time}/${file_flag}_vdl"

# train
rm -rf ${log_file}
rm -rf ${vdl}

python3.10 -m paddle.distributed.launch \
    --log_dir ${log_file} \
    --gpus 4,5,6,7 \
    run_ppo.py \
    --logging_dir ${vdl} \
    \
    --train_datasets Jsonfile::/root/paddlejob/workspace/gongenlei/3-7-pp_train-ppnlp.jsonl \
    --eval_datasets Jsonfile::/root/paddlejob/workspace/gongenlei/5ppl_test-ppnlp.jsonl \
    --actor_model_name_or_path ${actor_model_name_or_path} \
    --rl_algorithm reinforce_plus_plus \
    --use_rm_server 1 \
    --reward_server "http://127.0.0.1:8731" \
    \
    --do_train 1 \
    --learning_rate ${lr} \
    --lr_scheduler_type "constant" \
    --kl_coeff 0.001 \
    --max_grad_norm 1.0 \
    --output_dir "${current_time}/checkpoints/${file_flag}" \
    --max_steps ${max_steps} \
    --save_steps 200 \
    --save_strategy "steps" \
    --save_total_limit 10 \
    --ignore_save_lr_and_optim 1 \
    --seed 42 \
    --disable_tqdm 1 \
    --logging_steps 1 \
    --normalize_reward 0 \
    --normalize_advantage 0 \
    --sequence_parallel 0 \
    \
    --do_eval 1 \
    --per_device_eval_batch_size 32 \
    --evaluation_strategy "steps" \
    --eval_steps 20 \
    \
    --max_prompt_len 512 \
    --max_dec_len 4096 \
    --min_dec_len 1 \
    --repetition_penalty ${repetition_penalty} \
    --top_p ${top_p} \
    --temperature ${temperature} \
    --num_return_sequences 8 \
    --per_device_prompt_batch_size 8 \
    --per_device_rollout_batch_size 8 \
    \
    --max_length 4608 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --recompute 1 \
    --recompute_granularity "full" \
    --sharding_parallel_degree 1 \
    --sharding "stage1" \
    --tensor_parallel_degree 4 \
    --tensor_parallel_output 1 \
    --bf16 1 \
    --fp16_opt_level O2 \
    --offload_level "freeze_model" \
    --release_grads 1 \
    --report_to "visualdl" "wandb" \
    --run_name "reinforce_kl_no_entropy_no_amp" \
    --use_flash_attention 1 \
    --weight_decay 0.01 \
    --kl_loss_coeff 0.0 \
    --clip_range_score 10000 \
    --amp_master_grad 0