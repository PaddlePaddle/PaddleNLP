GPU_NUM=0
TASK_NAME=ernie
LOG_DIR=../log_$TASK_NAME

# Classification task
# Dataset: xnli, chnsenticorp
cd classification 
PYTHONPATH=../../../../../
python -m paddle.distributed.launch \
    --log_dir $LOG_DIR/xnli_log \
    --gpus $GPU_NUM \
    train.py \
    --dataset xnli \
    --batch_size 256\
    --learning_rate 1e-4\
    --epochs 3 \
    --valid_steps 100 \
    --save_steps 1e12 \
    --device gpu \
    --use_amp true\
    --save_dir ./checkpoints

PYTHONPATH=../../../../../ \
python -m paddle.distributed.launch \
    --log_dir  $LOG_DIR/chnsenticorp \
    --gpus $GPU_NUM \
    train.py \
    --device gpu \
    --learning_rate 5e-5\
    --batch_size 16 \
    --use_amp true\
    --epochs 8 \
    --valid_steps 100 \
    --save_steps 1e12 \
    --save_dir ./checkpoints

cd -

# NER tasks
# Dataset: peoples_daily_ner
cd ./ner
PYTHONPATH=../../../../../ \
python -m paddle.distributed.launch \
    --log_dir $LOG_DIR/peoples_daily_ner \
    --gpus $GPU_NUM \
    train.py \
    --model_type ernie \
    --model_name_or_path ernie-1.0 \
    --dataset peoples_daily_ner \
    --max_seq_length 128 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --num_train_epochs 8 \
    --logging_steps 10 \
    --save_steps 1e12 \
    --valid_steps 200 \
    --output_dir ./tmp/msra_ner/ \
    --device gpu

cd -


#  Dataset: CMRC
cd ./cmrc
PYTHONPATH=../../../../../ \
python -m paddle.distributed.launch \
    --log_dir $LOG_DIR/cmrc \
    --gpus $GPU_NUM \
    run_du.py \
    --task_name dureader_robust \
    --model_type ernie \
    --model_name_or_path ernie-1.0 \
    --max_seq_length 384 \
    --batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_steps 1000 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir ./tmp/dureader-robust_v2/ \
    --do_train \
    --do_predict \
    --device gpu
cd -
