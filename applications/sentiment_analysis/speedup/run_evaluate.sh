export CUDA_VISIBLE_DEVICES=1

python  evaluate.py \
        --base_model_path "../checkpoints/ppminilm" \
        --model_path "../checkpoints/sp_checkpoints/best.pdparams" \
        --test_path "../data/cls_data/test.txt" \
        --label_path "../data/cls_data/label.dict" \
        --batch_size 16 \
        --max_seq_len 256

