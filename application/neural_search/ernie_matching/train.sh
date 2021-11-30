python -u -m paddle.distributed.launch --gpus "0" train_pointwise.py \
        --device gpu \
        --save_dir ./checkpoints \
        --batch_size 32 \
        --learning_rate 2E-5