python -u -m paddle.distributed.launch --gpus '3' \
	train.py \
	--device gpu \
	--save_dir ./checkpoints/ \
	--batch_size 64 \
	--learning_rate 5E-5 \
	--epochs 1 \
	--save_steps 10 \
	--eval_steps 100 \
	--max_seq_length 128 \
	--dropout 0.1 \
    --output_emb_size 0 \
    --dup_rate 0.1 \
	--train_set_file "./data/train_unsupervised.txt"