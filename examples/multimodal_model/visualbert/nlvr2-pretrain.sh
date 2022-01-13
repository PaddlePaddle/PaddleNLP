export DATA_DIR=./X_NLVR/
export LOG_DIR=./logs/nlvr2-pre
unset CUDA_VISIBLE_DEVICES

python -m paddle.distributed.launch --gpus "4" --log_dir $LOG_DIR run_pretrain.py \
    --input_dir $DATA_DIR \
    --output_dir $LOG_DIR \
    --dataset nlvr2 \
    --model_type visualbert \
    --model_name_or_path visualbert-nlvr2-pre \
    --image_feature_type nlvr2_detectron_fix_144 \
    --train_batch_size 8 \
    --learning_rate 1e-5 \
    --num_train_epochs 1