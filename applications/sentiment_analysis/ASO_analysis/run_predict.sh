export CUDA_VISIBLE_DEVICES=0

python  predict.py \
        --ext_model_path "./checkpoints/ext_checkpoints/best.pdparams" \
        --cls_model_path "./checkpoints/cls_checkpoints/best.pdparams" \
        --test_path "./data/test.txt" \
        --ext_label_path "./data/ext_data/label.dict" \
        --cls_label_path "./data/cls_data/label.dict" \
        --save_path "./data/sentiment_results.json" \
        --batch_size 8 \
        --ext_max_seq_len 512 \
        --cls_max_seq_len 256

