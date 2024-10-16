
export FLAGS_set_to_1d=False 
export NVIDIA_TF32_OVERRIDE=0

export PYTHONPATH=../../:$PYTHONPATH
python -u  -m paddle.distributed.launch \
    --log_dir "./log_ppt" \
    run_pretrain.py \
    --model_name_or_path "meta-llama/Llama-2-13b" \
    --tokenizer_name_or_path "meta-llama/Llama-2-13b" \
    --input_dir "./pre-data" \
    --output_dir "./output" \
    --split 949,50,1 \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --per_device_eval_batch_size 1 \
    --use_flash_attention 0 \
    --use_fused_rms_norm 0 \
    --virtual_pp_degree 1 \
    --learning_rate 0.00001 \
    --min_learning_rate 0.000001 \
    --max_steps 2000 \
    --save_steps 2000 \
    --seed 100 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --dataloader_num_workers 1 \
    --eval_steps 1001 \
    --tensor_parallel_degree 8 \
    --disable_tqdm true \
    --continue_training 0 \
    --do_train \
    --device "gpu" \
    --enable_linear_fused_grad_add false \
    --fuse_attention_qkv false \
    --fuse_attention_ffn false \
    --use_fused_rope false \
    --recompute_use_reentrant true \
    --data_cache "./data_cache" \
    --bf16 \
    --fp16_opt_level "O2" \
    --amp_master_grad \
    --load_sharded_model true \
    --save_sharded_model true \
    --pipeline_parallel_degree 1 \
    --ignore_data_skip 0 \
    --force_reshard_pp true \
    --tensor_parallel_config "enable_mp_async_allreduce enable_mp_skip_c_identity" \
    --sequence_parallel 1 \
    --pipeline_parallel_config "disable_partial_send_recv" \
    --sharding "stage1" \
    --sharding_parallel_degree 1 
