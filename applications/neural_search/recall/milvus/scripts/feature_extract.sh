CUDA_VISIBLE_DEVICES=2 python feature_extract.py \
        --model_dir ./output \
        --model_name_or_path rocketqa-zh-base-query-encoder \
        --batch_size 512 \
        --corpus_file "milvus/milvus_data.csv" 

