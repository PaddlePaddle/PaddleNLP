CUDA_VISIBLE_DEVICES=2 python feature_extract.py \
        --model_dir ./output \
        --batch_size 256 \
        --corpus_file "milvus/milvus_data.csv" 