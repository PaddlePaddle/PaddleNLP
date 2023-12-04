cd ../../
task_name4="Static/hot_bybird_baseline/modole_pp4"
log_dir4="Static/hot_bybird_baseline/log_pp4"
task_name4_single="Static/addckpt/ckpt"


export PYTHONPATH="../../../PaddleNLP/"
export FLAGS_cudnn_deterministic=1
export FLAGS_embedding_deterministic=1

command4="python3.7 -u  -m paddle.distributed.launch \
    --gpus "3,4,5,6" \
    --log_dir ${log_dir4} \
    run_pretrain.py \
    --model_type "gpt" \
    --model_name_or_path ${task_name4_single} \
    --tokenizer_name_or_path gpt2-medium-en \
    --input_dir "./data" \
    --output_dir "$task_name4" \
    --pipeline_parallel_degree 2 \
    --sharding "stage1" \
    --sharding_parallel_degree 4 \
    --tensor_parallel_degree 2 \
    --split 949,50,1 \
    --max_seq_length 1024 \
    --per_device_train_batch_size 2 \
    --seed 1234 \
    --fuse_attention_qkv 0 \
    --use_flash_attention 1 \
    --attention_probs_dropout_prob 0.1 \
    --hidden_dropout_prob 0.1 \
    --bf16  \
    --fp16_opt_level "O2"  \
    --amp_master_grad 1 \
    --learning_rate 0.00001 \
    --min_learning_rate 0.000005 \
    --max_steps 500000 \
    --save_steps 50000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 0.0 \
    --logging_steps 1\
    --continue_training 1 \
    --dataloader_num_workers 1 \
    --eval_steps 1000 \
    --report_to "visualdl" \
    --disable_tqdm true \
    --recompute 0 \
    --gradient_accumulation_steps 16 \
    --do_train \
    --device "gpu"
"
$command4
