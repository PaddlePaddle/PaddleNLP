export DATA_DIR=./X_COCO/
export LOG_DIR=./logs/nlvr2-coco-pre
unset CUDA_VISIBLE_DEVICES

python -m paddle.distributed.launch --gpus "2" --log_dir $LOG_DIR run_pretrain.py \
    --input_dir $DATA_DIR \
    --output_dir $LOG_DIR \
    --dataset coco_captions \
    --model_type visualbert \
    --model_name_or_path visualbert-nlvr2-coco-pre \
    --image_feature_type coco_detectron_fix_144 \
    --train_batch_size 12 \
    --learning_rate 1e-5 \
    --num_train_epochs 1