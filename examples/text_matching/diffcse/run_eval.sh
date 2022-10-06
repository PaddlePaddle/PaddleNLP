gpu_ids=0
export CUDA_VISIBLE_DEVICES=${gpu_ids}

log_dir="log_eval"
python -u -m paddle.distributed.launch --gpus ${gpu_ids} --log_dir ${log_dir} \
    run_diffcse.py \
	--mode "eval" \
	--encoder_name "rocketqa-zh-dureader-query-encoder" \
	--max_seq_length "128" \
	--output_emb_size "32" \
	--eval_set_file "your dev_set path" \
	--ckpt_dir "./checkpoints/best" \
	--batch_size "32" \
	--seed "0" \
	--device "gpu"
