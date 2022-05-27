CUDA_VISIBLE_DEVICES=2 python feature_extract.py \
        --data_name train \
        --model_dir ./output \
        --output_dir data \
        --corpus_file "./data/train_data.txt" 

CUDA_VISIBLE_DEVICES=2 python feature_extract.py \
        --data_name val \
        --model_dir=./output \
        --output_dir data \
        --corpus_file "./data/val_data.txt" 