python run_train.py \
--model_name_or_path glm-2b \
--task_name cnn_dm \
--data_path ./data/cnn_dm \
--num_epochs 15 \
--learning_rate 3e-5 \
--warmup 0.06 \
--weight_decay 0.1 \
--label_smoothing 0.1 \
--save_steps 10000 \
--logging_steps 50 \
--eval_steps 1000 \
--src_seq_length 608 \
--tgt_seq_length 160 \
--min_tgt_length 55 \
--length_penalty 0.7 \
--no_repeat_ngram_size 3 \
--num_beams 5 \
--select_topk \
--eval_batch_size_per_device 4 \
--output_dir checkpoints \
--recompute \
--fp16 \
--overwrite \

--max_grad_norm 1.0 \
--lr_scheduler_type linear \



**NOT SURE**
cloze-eval
<!-- task-mask -->
<!-- --num_layers 36 \
--hidden_size 2048 \
--num_attention_heads 32 \
--max_position_embeddings 1024 \
--tokenizer_type GPT2BPETokenizer \
--load-pretrained CKPT/blocklm-2b-512 -->

--eval_iters 100 \ # 验证的时候，跑多少个 iterations

--deepspeed
--deepspeed_config
--finetune
--checkpoint_activations
--no-load_lr_scheduler


**HYPER PARAMS**
num_gpus_per_worker 8
num_workers 2
host_file_path "./hostfile"
mp_size 1
master_port $(shuf -n 1 -i 10000-65535)

OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
DISTRIBUTED_ARGS="${OPTIONS_NCCL} deepspeed --hostfile ${HOST_FILE_PATH} --master_port ${MASTER_PORT} --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER}"
