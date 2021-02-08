#!/bin/bash
export FLAGS_enable_parallel_graph=1
export FLAGS_sync_nccl_allreduce=1
export FLAGS_fraction_of_gpu_memory_to_use=0.95
TASK_NAME='emotion_detection'
DATA_PATH=./data/
VOCAB_PATH=./data/vocab.txt
CKPT_PATH=./save_models/textcnn
MODEL_PATH=./models/textcnn

# run_train on train.tsv and do_val on dev.tsv
train() {
    python run_classifier.py \
        --task_name ${TASK_NAME} \
        --use_cuda true \
        --do_train true \
        --do_val true \
        --batch_size 64 \
        --data_dir ${DATA_PATH} \
        --vocab_path ${VOCAB_PATH} \
        --output_dir ${CKPT_PATH} \
        --save_steps 200 \
        --validation_steps 200 \
        --epoch 10 \
        --lr 0.002 \
        --config_path ./config.json \
        --skip_steps 100 \
        --enable_ce
}

export CUDA_VISIBLE_DEVICES=0
train | python _ce.py
sleep 20
export CUDA_VISIBLE_DEVICES=0,1,2,3
train | python _ce.py
