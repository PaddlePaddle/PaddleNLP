export CUDA_VISIBLE_DEVICES=1

python  static_predict.py \
        --base_model_name "skep_ernie_1.0_large_ch" \
        --ext_model_path "./extraction/checkpoints/static/infer" \
        --cls_model_path "./classification/checkpoints/static/infer" \
        --ext_label_path "./extraction/data/label_ext.dict" \
        --cls_label_path "./classification/data/label_cls.dict" \
        --max_seq_len 256


