CUDA_VISIBLE_DEVICES=0 python utils/feature_extract.py \
        --data_name corpus \
        --model_dir ./output \
        --output_dir data \
        --corpus_file "./data/corpus.txt" 