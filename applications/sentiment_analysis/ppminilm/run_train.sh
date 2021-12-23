export CUDA_VISIBLE_DEVICES=1

python  train.py \
        --base_model_path "./checkpoints/ppminilm" \
        --train_path "./data/train_cls.txt" \
        --dev_path "./data/dev_cls.txt" \
        --label_path "./data/label_cls.dict" \
        --num_epochs 5 \
        --batch_size 16 \
        --max_seq_len 256 \
        --learning_rate 3e-5 \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
        --warmup_proportion 0.1 \
        --log_steps 50 \
        --eval_steps 100 \
        --seed 1000 \
        --device "gpu" \
        --checkpoints "./checkpoints"
