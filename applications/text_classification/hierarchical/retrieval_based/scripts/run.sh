CUDA_VISIBLE_DEVICES=0 python utils/feature_extract.py \
        --data_name label \
        --model_dir ./output \
        --output_dir data \
        --corpus_file "./data/label.txt" 

python utils/vector_insert.py \
                    --vector_path ./data/label_embedding.npy