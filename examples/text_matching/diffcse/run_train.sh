gpu_ids=0
export CUDA_VISIBLE_DEVICES=${gpu_ids}

log_dir="log_diffcse_ErG3D3"
python -u -m paddle.distributed.launch --gpus ${gpu_ids} --log_dir ${log_dir} \
	run_diffcse.py \
	--mode "train" \
	--encoder_name "rocketqa-zh-dureader-query-encoder" \
	--generator_name "ernie-3.0-base-zh" \
	--discriminator_name "ernie-3.0-base-zh" \
	--max_seq_length "128" \
	--output_emb_size "32" \
	--train_set_file "../dataset/100w/train_webdata.txt" \
	--eval_set_file "../dataset/100w/test_v1.txt" \
	--save_dir "./checkpoints_diffcse_ErG3D3" \
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
