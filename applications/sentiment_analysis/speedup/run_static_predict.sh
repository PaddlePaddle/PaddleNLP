export CUDA_VISIBLE_DEVICES=1

python  static_predict.py \
        --base_model_path "../checkpoints/ppminilm" \
        --model_path "../checkpoints/sp_checkpoints/quant/infer" \
        --test_path "../data/cls_data/test.txt" \
        --label_path "../data/cls_data/label.dict" \
        --num_epochs 10 \
        --batch_size 16 \
        --max_seq_len 256