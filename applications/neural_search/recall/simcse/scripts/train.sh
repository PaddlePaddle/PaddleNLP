# simcse gpu
python -u -m paddle.distributed.launch --gpus '0,1,2,3' \
	train.py \
	--device gpu \
	--save_dir ./checkpoints/ \
	--batch_size 64 \
	--learning_rate 5E-5 \
	--epochs 3 \
	--save_steps 2000 \
	--eval_steps 100 \
	--max_seq_length 64 \
	--infer_with_fc_pooler \
	--dropout 0.2 \
	--output_emb_size 256 \
	--train_set_file "./recall/train_unsupervised.csv" \
	--test_set_file "./recall/dev.csv" \
	--model_name_or_path "ernie-3.0-medium-zh"

# simcse cpu
# python 	train.py \
# 	--device cpu \
# 	--save_dir ./checkpoints/ \
# 	--batch_size 64 \
# 	--learning_rate 5E-5 \
# 	--epochs 3 \
# 	--save_steps 2000 \
# 	--eval_steps 100 \
# 	--max_seq_length 64 \
# 	--infer_with_fc_pooler \
# 	--dropout 0.2 \
#	--output_emb_size 256 \
# 	--train_set_file "./recall/train_unsupervised.csv" \
# 	--test_set_file "./recall/dev.csv" 
# 	--model_name_or_path "ernie-3.0-medium-zh"

# post training + simcse
# python -u -m paddle.distributed.launch --gpus '0,1,2,3' \
# 	train.py \
# 	--device gpu \
# 	--save_dir ./checkpoints/ \
# 	--batch_size 64 \
# 	--learning_rate 5E-5 \
# 	--epochs 3 \
# 	--save_steps 2000 \
# 	--eval_steps 100 \
# 	--max_seq_length 64 \
# 	--infer_with_fc_pooler \
# 	--dropout 0.2 \
#	--output_emb_size 256 \
# 	--train_set_file "./recall/train_unsupervised.csv" \
# 	--test_set_file "./recall/dev.csv" 
# 	--model_name_or_path "post_ernie"



