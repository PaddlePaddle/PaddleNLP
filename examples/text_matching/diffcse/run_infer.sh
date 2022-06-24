export CUDA_VISIBLE_DEVICES=0


python run_diffcse.py \
	--mode "infer" \
	--extractor_name "rocketqa-zh-dureader-query-encoder" \
	--generator_name "ernie-1.0" \
	--discriminator_name "ernie-1.0" \
	--max_seq_length "128" \
	--output_emb_size "32" \
	--infer_set_file "./data/test.txt" \
	--ckpt_dir "./checkpoints/checkpoint_10000" \
    --save_infer_path "./infer_result.txt" \
	--batch_size "8" \
	--dropout "0.1" \
	--margin "0.0" \
	--scale "20" \
	--dup_rate "0.0" \
	--seed "0" \
	--device "gpu"
