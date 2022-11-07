export CUDA_VISIBLE_DEVICES=0

python  performance_test.py \
        --base_model_name "ppminilm-6l-768h" \
        --model_path "../checkpoints/pp_checkpoints/quant/infer" \
        --test_path "../data/cls_data/test.txt" \
        --label_path "../data/cls_data/label.dict" \
        --num_epochs 10 \
        --batch_size 16 \
        --max_seq_len 256