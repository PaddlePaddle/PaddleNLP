export CUDA_VISIBLE_DEVICES=0

python  evaluate.py \
        --model_path "../checkpoints/cls_checkpoints/best.pdparams" \
        --test_path "../data/cls_data/test.txt" \
        --label_path "../data/cls_data/label.dict" \
        --batch_size 16 \
        --max_seq_len 256
