#if use gpu， you could specify gpu devices
export CUDA_VISIBLE_DEVICES=0 
#if use cpu, you could specify cpu nums
export CPU_NUM=10
python  train.py \
--train_path='data/train/sentence_file_*'  \
--test_path='data/dev/sentence_file_*'  \
--vocab_path data/vocabulary_min5k.txt \
--learning_rate 0.2 \
--use_gpu True \
--all_train_tokens 35479 \
--local True $@
