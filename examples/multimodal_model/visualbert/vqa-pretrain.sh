export DATA_DIR=./X_COCO/
export LOG_DIR=./logs/vqa-pre
unset CUDA_VISIBLE_DEVICES

python -m paddle.distributed.launch --gpus "3" --log_dir $LOG_DIR run_pretrain.py \
    --input_dir $DATA_DIR \
    --output_dir $LOG_DIR \
    --dataset vqa2 \
    --model_type visualbert \
    --model_name_or_path visualbert-vqa-pre \
    --image_feature_type coco_detectron_fix_100 \
    --train_batch_size 16 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --save_steps 5000