gpu_ids=0
export CUDA_VISIBLE_DEVICES=${gpu_ids}

log_dir="log_train"
python -u -m paddle.distributed.launch --gpus ${gpu_ids} --log_dir ${log_dir} \
	run_diffcse.py \
	--mode "train" \
	--encoder_name "rocketqa-zh-dureader-query-encoder" \
	--generator_name "ernie-3.0-base-zh" \
	--discriminator_name "ernie-3.0-base-zh" \
	--max_seq_length "128" \
	--output_emb_size "32" \
	--train_set_file "your train_set path" \
	--eval_set_file "your dev_set path" \
	--save_dir "./checkpoints" \
	--log_dir ${log_dir} \
	--save_steps "50000" \
	--eval_steps "1000" \
	--batch_size "32" \
	--epochs "3" \
	--mlm_probability "0.15" \
	--lambda_weight "0.15" \
	--learning_rate "3e-5" \
	--weight_decay "0.01" \
	--warmup_proportion "0.01" \
	--seed "0" \
	--device "gpu"
