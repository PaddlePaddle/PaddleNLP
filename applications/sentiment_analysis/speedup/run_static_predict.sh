export CUDA_VISIBLE_DEVICES=1

python  static_predict.py \
        --base_model_path "./checkpoints/ppminilm" \
        --model_path "./checkpoints/quant/infer" \
        --test_path "./data/test_cls.txt" \
        --label_path "./data/label_cls.dict" \
        --num_epochs 10 \
        --batch_size 16 \
        --max_seq_len 256