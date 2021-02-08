train() {
python  train.py \
--train_path='data/train/sentence_file_*'  \
--test_path='data/dev/sentence_file_*'  \
--vocab_path data/vocabulary_min5k.txt \
--learning_rate 0.2 \
--use_gpu True \
--all_train_tokens 35479 \
--max_epoch 10 \
--log_interval 5 \
--dev_interval 20 \
--local True $@ \
--enable_ce \
--shuffle false \
--random_seed 100
}

export CUDA_VISIBLE_DEVICES=0 
train | python _ce.py

export CUDA_VISIBLE_DEVICES=0,1,2,3 
train | python _ce.py
