export CUDA_VISIBLE_DEVICES=2

python  extraction/evaluate.py \
        --model_path "./checkpoints/best_ext.pdparams" \
        --test_path "./data/data_ext/test_ext.txt" \
        --label_path "./data/data_ext/label_ext.dict" \
        --batch_size 8 \
        --max_seq_len 512

