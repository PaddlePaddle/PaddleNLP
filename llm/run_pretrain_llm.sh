#!/bin/bash

# 设置环境变量
export PYTHONPATH=../:$PYTHONPATH
export FLAGS_call_stack_level=3
export NVIDIA_TF32_OVERRIDE=0
export FLAGS_cudnn_deterministic=True
export FLAGS_embedding_deterministic=1 

# 设置输出目录
task_name="llama_pretrain"
case_out_dir="output/${task_name}"
case_log_dir="output/${task_name}_log"

# 清理旧的输出目录
rm -rf $case_out_dir
rm -rf $case_log_dir

# 启动训练
python -u -m paddle.distributed.launch \
    --gpus "0,1,2,3" \
    --log_dir "$case_log_dir" \
    run_pretrain.py \
    --model_name_or_path "meta-llama/Llama-2-7b" \
    --tokenizer_name_or_path "meta-llama/Llama-2-7b" \
    --input_dir "./data" \
    --split "949,50,1" \
    --num_hidden_layers 4 \
    --output_dir "$case_out_dir" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 8 \
    --tensor_parallel_degree 4 \
    --pipeline_parallel_degree 1 \
    --tensor_parallel_config "enable_delay_scale_loss enable_mp_async_allreduce enable_mp_skip_c_identity" \
    --pipeline_parallel_config "enable_delay_scale_loss enable_release_grads disable_partial_send_recv enable_overlap_p2p_comm" \
    --virtual_pp_degree 1 \
    --sequence_parallel 0 \
    --use_flash_attention 0 \
    --use_fused_rms_norm 0 \
    --enable_linear_fused_grad_add 0 \
    --learning_rate 3e-05 \
    --logging_steps 1 \
    --max_steps 10 \
    --save_steps 11 \
    --eval_steps 1000 \
    --weight_decay 0.01 \
    --fp16 1 \
    --fp16_opt_level "O2" \
    --amp_master_grad 1 \
    --max_grad_norm 1.0 \
    --dataloader_num_workers 1 \
    --continue_training 0 \
    --do_train true \
    --do_eval false \
    --do_predict false \
    --disable_tqdm true \
    --skip_profile_timer true \
    --recompute 0 \
    --save_total_limit 2 \
    --device "gpu" \
    --save_sharded_model 0 \
    --unified_checkpoint 0 \
    --using_flex_checkpoint 1 \
    --fuse_attention_qkv true \
    --fuse_attention_ffn true \
    # --resume_from_checkpoint "./output/llama_pretrain/checkpoint-1"
