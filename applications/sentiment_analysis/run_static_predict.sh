export CUDA_VISIBLE_DEVICES=1

python  static_predict.py \
        --base_model_name "skep_ernie_1.0_large_ch" \
        --ext_model_path "./checkpoints/ext_checkpoints/static/infer" \
        --cls_model_path "./checkpoints/cls_checkpoints/static/infer" \
        --ext_label_path "./data/ext_data/label.dict" \
        --cls_label_path "./data/cls_data/label.dict" \
        --max_seq_len 256


