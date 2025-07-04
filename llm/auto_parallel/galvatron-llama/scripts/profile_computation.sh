set -x
unset CUDA_VISIBLE_DEVICES

task_name="llama_profile_computation"
rm -rf output/$task_name/
rm -rf "output/$task_name""_log"

export SOT_LOG_LEVEL=4
export PYTHONPATH=../../../:$PYTHONPATH

TRAINER="./train_dist_random.py"
LAUNCHER="python -u -m paddle.distributed.launch"
LAUNCHER="${LAUNCHER} --gpus 7"  # 设置需要使用的GPU
LAUNCHER="${LAUNCHER} --log_dir output/$task_name""_log ${TRAINER} --output_dir "./output""

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
    --num_hidden_layers 16 \
    --intermediate_size 11008 \
    --vocab_size 32000 \
    --hidden_size 4096 \
    --seq_length 1024 \
    --num_attention_heads 32 \
"

# [mbsz, accumulation_steps] [recompute] [amp]
CONFIG_ARGS="
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --recompute false \
    --recompute_use_reentrant true \
    --recompute_granularity full \
    --pp_recompute_interval 0 \
    --bf16 true \
    --fp16_opt_level "O1" \
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
    --sharding_parallel_config "enable_overlap"
    --tensor_parallel_config "enable_mp_async_allreduce"
)

# [fused] [flash_attention]
DEFAULT_OPTIMIZER="
    --fuse_attention_ffn true \
    --fuse_attention_qkv true \
    --fused_linear_param_grad_add 1 \
    --fuse_sequence_parallel_allreduce false \
    --use_flash_attention true \
    --use_fused_rope true \
    --use_fused_rms_norm true \
"

# [data]
DATA_ARGS="
    --input_dir ./data \
    --split 949,50,1 \
    --max_seq_length 1024"

# [runtime profiler]
RUNTIME_PROFILE_ARGS="
    --profile_time_flag 1 \
    --profile_forward_only 1 \
    --save_time_flag 1 \
"

# [model profiler]
MODEL_PROFILER_ARGS="
    --profile_type computation \
    --profile_mode batch \
    --profile_min_batch_size 1 \
    --profile_max_batch_size 12 \
    --profile_batch_size_step 1 \
    --layernum_min 2 \
    --layernum_max 4 \
    --profile_fixed_seq_length_list 1024 \
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