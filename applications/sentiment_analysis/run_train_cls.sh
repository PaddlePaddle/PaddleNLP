export CUDA_VISIBLE_DEVICES=6

python  classification/train.py \
        --train_path "./data/data_cls/train_cls.txt" \
        --dev_path "./data/data_cls/dev_cls.txt" \
        --label_path "./data/data_cls/label_cls.dict" \
        --num_epoch 3 \
        --batch_size 16 \
        --max_seq_len 256 \
        --learning_rate 3e-5 \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
        --warmup_proportion 0.1 \
        --log_step 50 \
        --eval_step 500 \
        --seed 1000 \
        --device "gpu" \
        --checkpoint "./checkpoints"
