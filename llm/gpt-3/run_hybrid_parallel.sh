#!/bin/sh
task_name1="mp2-pp2"
task_name2="mp2"

export PYTHONPATH="../../../PaddleNLP_PP_recompute/"
export FLAGS_cudnn_deterministic=True

max_steps=10
gradient_accumulation_steps=2
amp_master_grad=1
per_device_train_batch_size=2

log_dir1='mp2-pp2'
log_dir2='mp2'


rm -rf output/$task_name1 output/$task_name1
rm -rf $log_dir1 
rm -rf $log_dir2

export FLAGS_cudnn_deterministic=1
export FLAGS_embedding_deterministic=2

# export NVIDIA_TF32_OVERRIDE=0
# export FLAGS_embedding_deterministic=1
# export FLAGS_cudnn_deterministic=1
# export Flags_mp_aysnc_allreduce=1
# export Flags_skip_mp_c_identity=1
# export FLAGS_shard_norm_align_dp=0
# export FLAGS_shard_use_reduce=1


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
    --per_device_train_batch_size $per_device_train_batch_size \
    --seed 1234 \
    --fuse_attention_qkv 1 \
    --use_flash_attention 0 \
    --bf16  \
    --fp16_opt_level "O2"  \
    --amp_master_grad $amp_master_grad \
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
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --do_train \
    --attention_probs_dropout_prob 0 \
    --hidden_dropout_prob 0 \
    --max_grad_norm 0.0 \
    --device "gpu"
"
$command1

export FLAGS_cudnn_deterministic=1
export FLAGS_embedding_deterministic=2

command2="python3.7 -u -m paddle.distributed.launch \
    --gpus "0,1,2,3" \
    --log_dir ${log_dir2} \
    run_pretrain.py \
    --model_type "gpt" \
    --model_name_or_path output/hot_launch_ckpt/ \
    --tokenizer_name_or_path gpt2-medium-en \
    --input_dir "./data" \
    --output_dir "output/$task_name2" \
    --tensor_parallel_degree 2 \
    --sequence_parallel true \
    --split 949,50,1 \
    --max_seq_length 1024 \
    --per_device_train_batch_size $per_device_train_batch_size \
    --seed 1234 \
    --fuse_attention_qkv 1 \
    --use_flash_attention 0 \
    --bf16  \
    --fp16_opt_level "O2"  \
    --amp_master_grad $amp_master_grad \
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
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --do_train \
    --attention_probs_dropout_prob 0 \
    --hidden_dropout_prob 0 \
    --max_grad_norm 0.0 \
    --device "gpu"
"
$command2

# grep sequence_parallel mp2/workerlog.0
# grep sequence_parallel mp2-pp2/workerlog.0
grep loss mp2/workerlog.0
grep loss mp2-pp2/workerlog.0

# log_dir0='hot_launch_ckpt-hack'
# task_name0="hot_launch_ckpt-hack"

# command3="python3.7 -u  -m paddle.distributed.launch \
#     --gpus "3" \
#     --log_dir ${log_dir0} \
#     run_pretrain.py \
#     --model_type "gpt" \
#     --model_name_or_path gpt2-medium-en \
#     --tokenizer_name_or_path gpt2-medium-en \
#     --input_dir "./data" \
#     --output_dir "output/$task_name0" \
#     --split 949,50,1 \
#     --max_seq_length 1024 \
#     --per_device_train_batch_size 1 \
#     --seed 1234 \
#     --fuse_attention_qkv 1 \
#     --use_flash_attention 0 \
#     --bf16  \
#     --fp16_opt_level "O2"  \
#     --amp_master_grad 1 \
#     --learning_rate 0.00001 \
#     --min_learning_rate 0.000005 \
#     --max_steps 2 \
#     --save_steps 500000 \
#     --weight_decay 0.01 \
#     --warmup_ratio 0.01 \
#     --max_grad_norm 1.0 \
#     --logging_steps 1\
#     --continue_training 0 \
#     --dataloader_num_workers 1 \
#     --eval_steps 1000 \
#     --report_to "visualdl" \
#     --disable_tqdm true \
#     --recompute 0 \
#     --gradient_accumulation_steps 1 \
#     --do_train \
#     --attention_probs_dropout_prob 0 \
#     --hidden_dropout_prob 0 \
#     --max_grad_norm 0.0 \
#     --device "gpu"
# "
# $command3