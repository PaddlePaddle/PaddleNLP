export CUDA_VISIBLE_DEVICES=6

python  dynamic_predict.py \
        --ext_model_path "./extraction/checkpoints/best_ext.pdparams" \
        --cls_model_path "./classification/checkpoints/best_cls.pdparams" \
        --ext_label_path "./extraction/data/label_ext.dict" \
        --cls_label_path "./classification/data/label_cls.dict" \
        --max_seq_len 256

