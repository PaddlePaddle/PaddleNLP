python -u -m paddle.distributed.launch --gpus '4' \
	train.py \
	--device gpu \
	--save_dir ./checkpoints/ \
	--batch_size 64 \
	--learning_rate 5E-5 \
	--epochs 8 \
	--save_steps 2000 \
	--eval_steps 100 \
	--max_seq_length 64 \
	--dropout 0.3 \
    --output_emb_size 256 \
    --dup_rate 0.32 \
	--train_set_file "./senteval_cn/STS-B/train.txt" \
	--test_set_file "./senteval_cn/STS-B/dev.tsv" 