#!/bin/bash
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=0
MODEL_PATH=./pretrain_models/ernie
TASK_DATA_PATH=./data
CKPT_PATH=./save_models/ernie

# run_train
train() {
    python run_ernie_classifier.py \
        --use_cuda true \
        --verbose true \
        --do_train true \
        --do_val true \
        --use_paddle_hub false \
        --batch_size 32 \
        --init_checkpoint ${MODEL_PATH}/params \
        --train_set ${TASK_DATA_PATH}/train.tsv \
        --dev_set ${TASK_DATA_PATH}/dev.tsv \
        --vocab_path ${MODEL_PATH}/vocab.txt \
        --save_checkpoint_dir ${CKPT_PATH} \
        --save_steps 500 \
        --validation_steps 50 \
        --epoch 3 \
        --max_seq_len 64 \
        --ernie_config_path ${MODEL_PATH}/ernie_config.json \
        --lr 2e-5 \
        --skip_steps 50 \
        --num_labels 3 \
        --random_seed 1
}

# run_test
evaluate() {
    python run_ernie_classifier.py \
        --use_cuda true \
        --verbose true \
        --do_val true \
        --use_paddle_hub false \
        --batch_size 32 \
        --init_checkpoint ${CKPT_PATH}/step_907 \
        --test_set ${TASK_DATA_PATH}/test.tsv \
        --vocab_path ${MODEL_PATH}/vocab.txt \
        --max_seq_len 64 \
        --ernie_config_path ${MODEL_PATH}/ernie_config.json \
        --num_labels 3
}

# run_infer
infer() {
    python run_ernie_classifier.py \
        --use_cuda true \
        --verbose true \
        --do_infer true \
        --use_paddle_hub false \
        --batch_size 32 \
        --init_checkpoint ${CKPT_PATH}/step_907 \
        --infer_set ${TASK_DATA_PATH}/infer.tsv \
        --vocab_path ${MODEL_PATH}/vocab.txt \
        --max_seq_len 64 \
        --ernie_config_path ${MODEL_PATH}/ernie_config.json \
        --num_labels 3
}

main() {
    local cmd=${1:-help}
    case "${cmd}" in
        train)
            train "$@";
            ;;
        eval)
            evaluate "$@";
            ;;
        infer)
            infer "$@";
            ;;
        help)
            echo "Usage: ${BASH_SOURCE} {train|eval|infer}";
            return 0;
            ;;
        *)
            echo "unsupport command [${cmd}]";
            echo "Usage: ${BASH_SOURCE} {train|eval|infer}";
            return 1;
            ;;
    esac
}
main "$@"
