#!/bin/sh
task_name1="mp2-pp2-sp1-dp2"
task_name2="single_card"

export PYTHONPATH="../../../PaddleNLP_PP_recompute/"

max_steps=50000
log_dir1='p2-pp2-sp1-dp2'
log_dir2='single_card'


rm -rf output/$task_name1 output/$task_name1
rm -rf $log_dir1 
rm -rf $log_dir2

export FLAGS_cudnn_deterministic=1
export FLAGS_embedding_deterministic=1

command1="python3.7 -u -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir ${log_dir1} \
    run_pretrain.py \
    --model_type "gpt" \
    --model_name_or_path output/hot_launch_ckpt \
    --tokenizer_name_or_path gpt2-medium-en \
    --input_dir "./data" \
    --output_dir "output/$task_name1" \
    --tensor_parallel_degree 2 \
    --pipeline_parallel_degree 2 \
    --pipeline_parallel_config "disable_partial_send_recv" \
    --sequence_parallel true \
    --split 949,50,1 \
    --max_seq_length 1024 \
    --per_device_train_batch_size 8 \
    --seed 1234 \
    --fuse_attention_qkv 1 \
    --use_flash_attention 0 \
    --bf16  \
    --fp16_opt_level "O2"  \
    --amp_master_grad 1 \
    --learning_rate 0.00001 \
    --min_learning_rate 0.000005 \
    --max_steps $max_steps \
    --save_steps 500000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --logging_steps 1 \
    --continue_training 1 \
    --dataloader_num_workers 1 \
    --eval_steps 1000 \
    --report_to "visualdl" \
    --disable_tqdm true \
    --recompute 0 \
    --gradient_accumulation_steps 2 \
    --do_train \
    --attention_probs_dropout_prob 0 \
    --hidden_dropout_prob 0 \
    --max_grad_norm 0.0 \
    --device "gpu"
"
$command1

export FLAGS_cudnn_deterministic=1
export FLAGS_embedding_deterministic=1

command2="python3.7 -u -m paddle.distributed.launch \
    --gpus "0" \
    --log_dir ${log_dir2} \
    run_pretrain.py \
    --model_type "gpt" \
    --model_name_or_path output/hot_launch_ckpt/ \
    --tokenizer_name_or_path gpt2-medium-en \
    --input_dir "./data" \
    --output_dir "output/$task_name2" \
    --split 949,50,1 \
    --max_seq_length 1024 \
    --per_device_train_batch_size 16 \
    --seed 1234 \
    --fuse_attention_qkv 1 \
    --use_flash_attention 0 \
    --bf16  \
    --fp16_opt_level "O2"  \
    --amp_master_grad 1 \
    --learning_rate 0.00001 \
    --min_learning_rate 0.000005 \
    --max_steps $max_steps \
    --save_steps 500000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --logging_steps 1 \
    --continue_training 1 \
    --dataloader_num_workers 1 \
    --eval_steps 1000 \
    --report_to "visualdl" \
    --disable_tqdm true \
    --recompute 0 \
    --gradient_accumulation_steps 2 \
    --do_train \
    --attention_probs_dropout_prob 0 \
    --hidden_dropout_prob 0 \
    --max_grad_norm 0.0 \
    --device "gpu"
"
$command2