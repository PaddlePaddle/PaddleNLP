export TASK_NAME=$1
export LR=$2
export BS=$3
export EPOCH=$4
export MAX_SEQ_LEN=$5
export CUDA_VISIBLE_DEVICES=$6
export MODEL_PATH=$7

git rev-parse HEAD


#python -m paddle.distributed.launch --gpus "0,1,2" run_clue_classifier.py \
python -u ./run_clue_classifier.py \
    --model_name_or_path ${MODEL_PATH} \
    --task_name ${TASK_NAME} \
    --max_seq_length ${MAX_SEQ_LEN} \
    --batch_size ${BS}   \
    --learning_rate ${LR} \
    --num_train_epochs ${EPOCH} \
    --logging_steps 100 \
    --seed 42  \
    --save_steps  100 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-8 \
    --output_dir ${MODEL_PATH}/models/${TASK_NAME}/${LR}_${BS}/ \
    --device gpu  \
    --gradient_accumulation_steps $8 \
    --do_train \
    --dropout $9 \
