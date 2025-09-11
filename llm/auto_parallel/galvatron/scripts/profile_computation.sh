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

set -x
unset CUDA_VISIBLE_DEVICES

unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
unset PADDLE_TRAINERS_NUM
unset PADDLE_TRAINER_ID
unset PADDLE_WORKERS_IP_PORT_LIST
unset PADDLE_TRAINERS
unset PADDLE_NUM_GRADIENT_SERVERS

source <path_to_your_own_python>

task_name="qwen"
dir_name="profile_computation"
rm -rf output/$dir_name/$task_name/
rm -rf "output/$dir_name/$task_name""_log"

export SOT_LOG_LEVEL=4
export PYTHONPATH=../../../:$PYTHONPATH

TRAINER="./train_qwen.py"
LAUNCHER="python -u -m paddle.distributed.launch"
LAUNCHER="${LAUNCHER} --gpus 7"  # 设置需要使用的GPU
LAUNCHER="${LAUNCHER} --log_dir output/$dir_name/$task_name""_log ${TRAINER} --output_dir "./output""

export LAUNCHER=$LAUNCHER

# [max_steps] [logging_steps] [enable_auto_parallel]
TRAIN_ARGS="
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --learning_rate 3e-05 \
    --min_learning_rate 3e-06 \
    --max_steps 25 \
    --logging_steps 1 \
    --continue_training 0 \
    --do_train true \
    --do_eval false \
    --do_predict false \
    --disable_tqdm true \
    --skip_profile_timer false \
    --skip_memory_metrics 0 \
    --save_total_limit 2 \
    --device gpu \
    --dataloader_num_workers 1 \
    --distributed_dataloader 0 \
    --enable_auto_parallel 1 \
"

# [seq_length] [num_hidden_layers]
MODEL_ARGS="
    --model_name_or_path "llama" \
    --tokenizer_name_or_path "llama" \
    --num_hidden_layers 2 \
    --intermediate_size 25600 \
    --vocab_size 32000 \
    --hidden_size 5120 \
    --seq_length 1024 \
    --num_attention_heads 64 \
    --num_key_value_heads 8 \
"

# [mbsz, accumulation_steps] [recompute] [amp]
CONFIG_ARGS="
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --recompute true \
    --recompute_use_reentrant true \
    --recompute_granularity full \
    --pp_recompute_interval 0 \
    --bf16 true \
    --fp16_opt_level "O2" \
    --amp_master_grad true \
    --amp_custom_black_list "reduce_sum" "c_softmax_with_cross_entropy" \
    --amp_custom_white_list "lookup_table" "lookup_table_v2" \
"

# [dp_deg, dp_type] [tp_deg, megatron-sp] [pp_deg, 1F1B] [parallel_configs]
PARALLEL_ARGS=(
    --to_static 0
    --sharding_parallel_degree 1
    --sharding "stage2"
    --tensor_parallel_degree 2
    --sequence_parallel true
    --pipeline_parallel_degree 2
    --virtual_pp_degree 1
    --pipeline_schedule_mode "1F1B"
    --sep_parallel_degree 1
    --pipeline_parallel_config "enable_send_recv_overlap"
    --data_parallel_config "enable_allreduce_avg_in_gradinent_scale gradient_sync_after_accumulate"
    --sharding_parallel_config "enable_overlap enable_release_grads"
    --tensor_parallel_config "enable_mp_async_allreduce replace_with_parallel_cross_entropy"
)

# [fused] [flash_attention]
DEFAULT_OPTIMIZER="
    --fuse_attention_ffn true \
    --fuse_attention_qkv true \
    --fused_linear_param_grad_add 1 \
    --fuse_sequence_parallel_allreduce true \
    --use_flash_attention true \
    --use_fused_rope true \
    --use_fused_rms_norm false \
    --enable_linear_fused_grad_add true \
"

# [data]
DATA_ARGS="
    --input_dir ./data \
    --split 949,50,1 \
    --max_seq_length 16384"

# [runtime profiler]
RUNTIME_PROFILE_ARGS="
    --profile_time_flag 1 \
    --profile_forward_only 1 \
    --save_time_flag 1 \
"

# [model profiler] [sequence type]
MODEL_PROFILER_ARGS="
    --profile_type computation \
    --profile_mode sequence \
    --profile_fixed_batch_size 1 \
    --layernum_min 1 \
    --layernum_max 2 \
    --profile_min_seq_length 4096 \
    --profile_max_seq_length 16384 \
    --profile_seq_length_step 4096 \
    --num_layertype 1 \
"

python ./profile.py \
    $MODEL_ARGS \
    $TRAIN_ARGS \
    $CONFIG_ARGS \
    "${PARALLEL_ARGS[@]}" \
    $DEFAULT_OPTIMIZER \
    $DATA_ARGS \
    $RUNTIME_PROFILE_ARGS \
    $MODEL_PROFILER_ARGS