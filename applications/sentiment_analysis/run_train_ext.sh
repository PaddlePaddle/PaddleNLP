export CUDA_VISIBLE_DEVICES=2

python  extraction/train.py \
        --train_path "./data/data_ext/train_ext.txt" \
        --dev_path "./data/data_ext/dev_ext.txt" \
        --label_path "./data/data_ext/label_ext.dict" \
        --num_epoch 10 \
        --batch_size 8 \
        --max_seq_len 512 \
        --learning_rate 5e-5 \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
        --warmup_proportion 0.1 \
        --log_step 50 \
        --eval_step 1000 \
        --seed 1000 \
        --device "gpu" \
        --checkpoint "./checkpoints"
