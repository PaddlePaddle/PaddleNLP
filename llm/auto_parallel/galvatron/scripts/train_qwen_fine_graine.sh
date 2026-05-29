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

nnodes=$PADDLE_TRAINERS_NUM
rank=$PADDLE_TRAINER_ID

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

START_RANK=0
END_RANK=8

if [[ $rank -lt $START_RANK ]]; then
    exit 0
fi

if [[ $rank -ge $END_RANK ]]; then
    exit 0
fi
export rank=$(($rank-$START_RANK))
export nnodes=$(($END_RANK-$START_RANK))
master_ip=`cat /root/paddlejob/workspace/hostfile | head -n $(($START_RANK+1)) | tail -n 1 | awk '{print $1}'`
export master=$master_ip
export port=36677

export interpreter="<path to your own python>"

task_name="fine_grained_config-with-manual"
dir_name="fine-vs-corase"

rm -rf output/$dir_name/$task_name/
rm -rf "output/$dir_name/$task_name""_log"

export SOT_LOG_LEVEL=4
export PYTHONPATH=../../../:$PYTHONPATH

TRAINER="./train_qwen_fine_graine.py"
LAUNCHER="python -u -m paddle.distributed.launch"
LAUNCHER="${LAUNCHER} --master $master:$port --nnodes $nnodes --rank $rank --gpus 0,1,2,3,4,5,6,7"
LAUNCHER="${LAUNCHER} --log_dir output/$dir_name/$task_name""_log ${TRAINER} --output_dir "./output""

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
# still need to use llama as model_type
MODEL_ARGS=(
    --model_type "llama_fine_grained_final"
    --num_hidden_layers 72
    --intermediate_size 25600
    --vocab_size 32000
    --hidden_size 5120
    --seq_length 32768
    --num_attention_heads 64
    --num_key_value_heads 8
)

# "max_position_embeddings": 32768,
# [mbsz, accumulation_steps] [recompute] [amp]
CONFIG_ARGS="
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --recompute false \
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
    --to_static 1
    --sharding_parallel_degree 2
    --sharding "stage2"
    --tensor_parallel_degree 4
    --sequence_parallel true
    --pipeline_parallel_degree 1
    --virtual_pp_degree 1
    --pipeline_schedule_mode "1F1B"
    --sep_parallel_degree 1
    --pipeline_parallel_config "enable_send_recv_overlap"
    --data_parallel_config "enable_allreduce_avg_in_gradinent_scale gradient_sync_after_accumulate"
    --sharding_parallel_config "enable_overlap enable_release_grads"
    --tensor_parallel_config "enable_mp_async_allreduce replace_with_parallel_cross_entropy"
)
#     --sharding_parallel_config "enable_overlap enable_release_grads enable_tensor_fusion"


# [fused] [flash_attention]
DEFAULT_OPTIMIZER_ARGS="
    --fuse_attention_ffn true \
    --fuse_attention_qkv true \
    --fused_linear_param_grad_add 1 \
    --fuse_sequence_parallel_allreduce true \
    --use_flash_attention true \
    --use_fused_rope true \
    --use_fused_rms_norm true \
    --enable_linear_fused_grad_add true \
"

# [data] max_seq_length equal config.max_position_embeddings
DATA_ARGS="
    --input_dir ./data \
    --split 949,50,1 \
    --max_seq_length 32768"

# [runtime_profile]
RUNTIME_PROFILE_ARGS="
    --profile_time_flag 1 \
    --profile_memory_flag 1 \
    --profile_forward_only 0 \
    --save_time_flag 0 \
    --save_memory_flag 0 \
"

# [debug] 
DEBUG_ARGS="
    --job_schedule_profiler_start 1 \
    --job_schedule_profiler_end 5 \
"   

# [GranularityRuntime]
GRANULARITY_RUNTIME_ARGS="
    --granularity_type fine_grained \
    --usp_flag 0 \
    --sharding_stage_level 2 \
    --fine_grained_config_path ./configs/fine_grained_config.json \
"

bash kill.sh
sleep 1

$LAUNCHER \
    "${MODEL_ARGS[@]}" \
    $TRAIN_ARGS \
    $CONFIG_ARGS \
    "${PARALLEL_ARGS[@]}" \
    $DEFAULT_OPTIMIZER_ARGS \
    $DATA_ARGS \
    $RUNTIME_PROFILE_ARGS \
    $GRANULARITY_RUNTIME_ARGS