export PYTHONPATH="../../../":$PYTHONPATH
export FLAGS_cudnn_deterministic=1
export FLAGS_embedding_deterministic=1 
export NVIDIA_TF32_OVERRIDE=0
export FLAGS_call_stack_level=3

to_static=0
# export TRANSLATOR_DISABLE_NEW_ERROR=0
# export TRANSLATOR_CODE_LEVEL=100

task_name="gpt3_auto_dp2mp2pp2_${to_static}"
log_dir="output/$task_name""_log"
output_dir="output/$task_name"
rm -rf $log_dir
rm -rf $output_dir

python -u -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir ${log_dir} \
    run_pretrain_auto.py \
    --model_name_or_path gpt2-medium-en \
    --tokenizer_name_or_path gpt2-medium-en \
    --input_dir "../data" \
    --output_dir ${output_dir}  \
    --split 949,50,1 \
    --max_seq_length 1024 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --sharding "" \
    --tensor_parallel_degree 2 \
    --pipeline_parallel_degree 2 \
    --sequence_parallel 0 \
    --fuse_attention_qkv 0 \
    --use_flash_attention 0 \
    --scale_loss 1024 \
    --learning_rate 0.00001 \
    --min_learning_rate 0.000005 \
    --max_steps 10 \
    --save_steps 50000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 1\
    --continue_training 0\
    --dataloader_num_workers 1 \
    --eval_steps 100000 \
    --report_to "visualdl" \
    --disable_tqdm true \
    --recompute 0 \
    --gradient_accumulation_steps 4 \
    --do_train \
    --do_eval \
    --device "gpu" \
    --model_type "gpt" \
    --enable_auto_parallel 1 \
    --to_static ${to_static} \
    --fp16 0 \
    --fp16_opt_level "O2"  \
