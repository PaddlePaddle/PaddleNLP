export CUDA_VISIBLE_DEVICES=0

python  train.py \
        --train_path "../data/cls_data/train.txt" \
        --dev_path "../data/cls_data/dev.txt" \
        --label_path "../data/cls_data/label.dict" \
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
        --checkpoints "../checkpoints/cls_checkpoints"
