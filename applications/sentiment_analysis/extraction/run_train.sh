export CUDA_VISIBLE_DEVICES=0

python  train.py \
        --train_path "./data/train_ext.txt" \
        --dev_path "./data/dev_ext.txt" \
        --label_path "./data/label_ext.dict" \
        --num_epochs 10 \
        --batch_size 16 \
        --max_seq_len 256 \
        --learning_rate 5e-5 \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
        --warmup_proportion 0.1 \
        --log_steps 50 \
        --eval_steps 250 \
        --seed 1000 \
        --device "gpu" \
        --checkpoints "./checkpoints"
