export DATA_DIR=./X_COCO/
export LOG_DIR=./logs/vqa
unset CUDA_VISIBLE_DEVICES

python -m paddle.distributed.launch --gpus "0" --log_dir $LOG_DIR VQA2/run_vqa2.py \
    --input_dir $DATA_DIR \
    --output_dir $LOG_DIR \
    --task_name vqa2 \
    --model_type visualbert \
    --model_name_or_path checkpoint/paddle_visualbert/visualbert-vqa-pre \
    --batch_size 64 \
    --learning_rate 2e-5 \
    --save_steps 10000 \
    --num_train_epochs 10 