set -x
unset CUDA_VISIBLE_DEVICES

task_name="qwen_profile_memory"
dir_name="profile_memory"
rm -rf output/$dir_name/$task_name/
rm -rf "output/$dir_name/$task_name""_log"

export SOT_LOG_LEVEL=4
export PYTHONPATH=../../../:$PYTHONPATH

TRAINER="./train_qwen.py"
LAUNCHER="/apdcephfs_fsgm/share_303760348/anaconda3/envs/lgm-paddle/bin/python -u -m paddle.distributed.launch"
LAUNCHER="${LAUNCHER} --gpus 0,1,2,3,4,5,6,7"  # 设置需要使用的GPU
LAUNCHER="${LAUNCHER} --log_dir output/$dir_name/$task_name""_log ${TRAINER} --output_dir "./output""

export LAUNCHER=$LAUNCHER
export PROFILE_WORLD_SIZE=8

# [max_steps] [logging_steps] [enable_auto_parallel]
TRAIN_ARGS="
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --learning_rate 3e-05 \
    --min_learning_rate 3e-06 \
    --max_steps 10 \
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
    --intermediate_size 25600 \
    --vocab_size 32000 \
    --hidden_size 5120 \
    --seq_length 1024 \
    --num_attention_heads 64 \
    --num_key_value_heads 8 \
"

# [mbsz, accumulation_steps] [recompute] [amp]
CONFIG_ARGS="
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
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

# [data]
DATA_ARGS="
    --input_dir ./data \
    --split 949,50,1 \
    --max_seq_length 32768"

# [runtime profiler]
RUNTIME_PROFILE_ARGS="
    --profile_memory_flag 1 \
    --save_memory_flag 1 \
"

# [model profiler] [static type]
MODEL_PROFILER_ARGS="
    --profile_type memory \
    --profile_mode static \
    --profile_fixed_batch_size 8 \
    --layernum_min 1 \
    --layernum_max 2 \
    --profile_fixed_seq_length_list 32768 \
    --num_layertype 1 \
    --max_tp_deg 8 \
    --max_per_device_train_batch_size 4 \
"

# [model profiler] [sequence type]
# MODEL_PROFILER_ARGS="
#     --profile_type memory \
#     --profile_mode sequence \
#     --profile_fixed_batch_size 8 \
#     --profile_min_seq_length 2048 \
#     --profile_max_seq_length 4096 \
#     --layernum_min 1 \
#     --layernum_max 2 \
#     --num_layertype 1 \
#     --max_tp_deg 8 \
# "

/apdcephfs_fsgm/share_303760348/anaconda3/envs/lgm-paddle/bin/python ./profile.py \
    $MODEL_ARGS \
    $TRAIN_ARGS \
    $CONFIG_ARGS \
    "${PARALLEL_ARGS[@]}" \
    $DEFAULT_OPTIMIZER_ARGS \
    $DATA_ARGS \
    $RUNTIME_PROFILE_ARGS \
    $MODEL_PROFILER_ARGS
    