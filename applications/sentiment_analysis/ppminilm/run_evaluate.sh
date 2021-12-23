export CUDA_VISIBLE_DEVICES=1

python  evaluate.py \
        --base_model_path "./checkpoints/ppminilm" \
        --model_path "./checkpoints/best_mini.pdparams" \
        --test_path "./data/test_cls.txt" \
        --label_path "./data/label_cls.dict" \
        --batch_size 16 \
        --max_seq_len 256

