gpu_ids=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=${gpu_ids}


python -u -m paddle.distributed.launch --gpus ${gpu_ids} \
	run_diffcse.py \
	--mode "train" \
	--extractor_name "rocketqa-zh-dureader-query-encoder" \
	--generator_name "ernie-1.0" \
	--discriminator_name "ernie-1.0" \
	--max_seq_length "128" \
	--output_emb_size "32" \
	--train_set_file ./data/train.txt \
	--eval_set_file ./data/test_v1.txt \
	--save_dir ./checkpoints \
	--save_steps "10000" \
	--eval_steps "1000" \
	--batch_size "32" \
	--epochs "5" \
	--learning_rate "3e-5" \
	--weight_decay "0.01" \
	--warmup_proportion "0.01" \
	--dropout "0.1" \
	--margin "0.0" \
	--scale "20" \
	--dup_rate "0.0" \
	--seed "0" \
	--device "gpu"