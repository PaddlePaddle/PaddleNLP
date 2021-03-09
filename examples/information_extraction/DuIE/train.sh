set -eux

export BATCH_SIZE=8
export LR=2e-5
export EPOCH=12

CUDA_VISIBLE_DEVICES=3 python run_duie.py \
                   --n_gpu 1 \
                   --seed 42 \
                   --do_train \
                   --data_path ./data \
                   --max_seq_length 128 \
                   --batch_size $BATCH_SIZE \
                   --num_train_epochs $EPOCH \
                   --learning_rate $LR \
                   --warmup_ratio 0.06 \
                   --output_dir ./checkpoints
