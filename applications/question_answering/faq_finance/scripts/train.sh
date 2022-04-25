python -u -m paddle.distributed.launch --gpus '0' \
	train.py \
	--device gpu \
	--save_dir ./checkpoints/ \
	--batch_size 64 \
	--learning_rate 5E-5 \
	--epochs 3 \
	--save_steps 50 \
	--eval_steps 50 \
	--max_seq_length 64 \
	--dropout 0.2 \
	--output_emb_size 256 \
	--dup_rate 0.1 \
	--rdrop_coef 0.1 \
	--train_set_file "./data/train_aug.csv" 