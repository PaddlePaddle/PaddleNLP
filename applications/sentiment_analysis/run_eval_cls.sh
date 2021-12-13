export CUDA_VISIBLE_DEVICES=2

python  classification/evaluate.py \
        --model_path "./checkpoints/best_cls.pdparams" \
        --test_path "./data/data_cls/test_cls.txt" \
        --label_path "./data/data_cls/label_cls.dict" \
        --batch_size 16 \
        --max_seq_len 256
