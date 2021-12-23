export CUDA_VISIBLE_DEVICES=1

python  evaluate.py \
        --model_path "./checkpoints/best_ext.pdparams" \
        --test_path "./data/test_ext.txt" \
        --label_path "./data/label_ext.dict" \
        --batch_size 16 \
        --max_seq_len 256

