export CUDA_VISIBLE_DEVICES=0

python  predict.py \
        --base_model_name "skep_ernie_1.0_large_ch" \
        --ext_model_path "../checkpoints/ext_checkpoints/static/infer" \
        --cls_model_path "../checkpoints/cls_checkpoints/static/infer" \
        --ext_label_path "../data/ext_data/label.dict" \
        --cls_label_path "../data/cls_data/label.dict" \
        --test_path "../data/test.txt" \
        --save_path "../data/sentiment_results.json" \
        --batch_size 8 \
        --ext_max_seq_len 512 \
        --cls_max_seq_len 256
