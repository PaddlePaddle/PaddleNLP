
# Cuda device ids used
cuda_device=""
# Folder used to save checkpoints
save_dir=
# Data set path for training
train_path=
# Data set path for development
dev_path=

###############################################################################################################################################################################################################################################################################################################################################
python -u -m paddle.distributed.launch --gpus $cuda_device finetune.py \
            --train_path $train_path \
            --dev_path $dev_path \
            --save_dir $save_dir \
            --learning_rate 1e-5 \
            --batch_size 32 \
            --max_seq_len 512 \
            --num_epochs 30 \
            --model uie-base \
            --seed 1000 \
            --logging_steps 100 \
            --valid_steps 5000 \
            --device gpu