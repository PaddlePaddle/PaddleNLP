CUDA_VISIBLE_DEVICES=0 python utils/feature_extract.py \
        --data_name train \
        --model_dir ./output \
        --output_dir data \
        --corpus_file "./data/corpus.txt" 

python utils/vector_insert.py \
                    --vector_path ./data/corpus_embedding.npy