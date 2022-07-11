gpu_ids=0
export CUDA_VISIBLE_DEVICES=${gpu_ids}

log_dir="log_infer"
python -u -m paddle.distributed.launch --gpus ${gpu_ids} --log_dir ${log_dir} \
    run_diffcse.py \
	--mode "infer" \
	--encoder_name "rocketqa-zh-dureader-query-encoder" \
	--max_seq_length "128" \
	--output_emb_size "32" \
	--infer_set_file "../diffcse/data/test.txt" \
	--ckpt_dir "./checkpoints_diffcse_ErG3D3/best_spearman" \
    --save_infer_path "./infer_result.txt" \
	--batch_size "8" \
	--seed "0" \
	--device "gpu"
