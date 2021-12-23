export CUDA_VISIBLE_DEVICES=7

python  dynamic_predict_by_batch.py \
        --ext_model_path "./extraction/checkpoints/best_ext.pdparams" \
        --cls_model_path "./classification/checkpoints/best_cls.pdparams" \
        --test_path "./data/test.txt" \
        --ext_label_path "./extraction/data/label_ext.dict" \
        --cls_label_path "./classification/data/label_cls.dict" \
        --save_path "./data/sentiment_results.json" \
        --batch_size 16 \
        --max_seq_len 256

