export CUDA_VISIBLE_DEVICES=6

python  predict.py \
        --ext_model_path "./checkpoints/best_ext.pdparams" \
        --cls_model_path "./checkpoints/best_cls.pdparams" \
        --ext_label_path "./data/data_ext/label_ext.dict" \
        --cls_label_path "./data/data_cls/label_cls.dict" \
        --max_seq_len 256

