#!/bin/sh
task_name1="hybrid_parallel_test1"
task_name2="hybrid_parallel_test2"

export PYTHONPATH="../../../PaddleNLP_PP_recompute/"
export FLAGS_cudnn_deterministic=True

log_dir1='hybrid_test1_main_grad_acc2'
log_dir2='hybrid_test2_main_grad_acc2'

rm -rf $task_name1 $task_name1
rm -rf $log_dir1 
rm -rf $log_dir2


export FLAGS_cudnn_deterministic=1
export FLAGS_embedding_deterministic=1

command1="python3.7 -u -m paddle.distributed.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir ${log_dir1} \
    run_pretrain.py \
    --model_type "gpt" \
    --model_name_or_path gpt2-medium-en \
    --tokenizer_name_or_path gpt2-medium-en \
    --input_dir "./data" \
    --output_dir "output/$task_name1" \
    --tensor_parallel_degree 2 \
    --pipeline_parallel_degree 2 \
    --sequence_parallel true \
    --split 949,50,1 \
    --max_seq_length 1024 \
    --per_device_train_batch_size 2 \
    --pipeline_parallel_config "disable_partial_send_recv" \
    --seed 1234 \
    --fuse_attention_qkv 1 \
    --use_flash_attention 0 \
    --bf16  \
    --fp16_opt_level "O2"  \
    --amp_master_grad 1 \
    --learning_rate 0.00001 \
    --min_learning_rate 0.000005 \
    --max_steps 50000 \
    --save_steps 500000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --continue_training 0 \
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

command2="python3.7 -u  -m paddle.distributed.launch \
    --gpus "3" \
    --log_dir ${log_dir2} \
    run_pretrain.py \
    --model_type "gpt" \
    --model_name_or_path gpt2-medium-en \
    --tokenizer_name_or_path gpt2-medium-en \
    --input_dir "./data" \
    --output_dir "output/$task_name2" \
    --split 949,50,1 \
    --max_seq_length 1024 \
    --per_device_train_batch_size 4 \
    --seed 1234 \
    --fuse_attention_qkv 1 \
    --use_flash_attention 0 \
    --bf16  \
    --fp16_opt_level "O2"  \
    --amp_master_grad 1 \
    --learning_rate 0.00001 \
    --min_learning_rate 0.000005 \
    --max_steps 50000 \
    --save_steps 500000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 1\
    --continue_training 0 \
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
# $command2