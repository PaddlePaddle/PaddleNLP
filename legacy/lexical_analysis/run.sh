#!/bin/bash
export FLAGS_fraction_of_gpu_memory_to_use=0.02
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fast_eager_deletion_mode=1
export CUDA_VISIBLE_DEVICES=0,1,2,3     #   which GPU to use

function run_train() {
    echo "training"
    python train.py \
        --train_data ./data/train.tsv \
        --test_data ./data/test.tsv \
        --model_save_dir ./models \
        --validation_steps 2 \
        --save_steps 10 \
        --print_steps 1 \
        --batch_size 300 \
        --epoch 10 \
        --traindata_shuffle_buffer 20000 \
        --word_emb_dim 128 \
        --grnn_hidden_dim 128 \
        --bigru_num 2 \
        --base_learning_rate 1e-3 \
        --emb_learning_rate 2 \
        --crf_learning_rate 0.2 \
        --word_dict_path ./conf/word.dic \
        --label_dict_path ./conf/tag.dic \
        --word_rep_dict_path ./conf/q2b.dic \
        --enable_ce false \
        --use_cuda false \
        --cpu_num 1
}

function run_train_single_gpu() {
    echo "single gpu training"              # which GPU to use
    export CUDA_VISIBLE_DEVICES=0
    python train.py \
        --use_cuda true
}

function run_train_multi_gpu() {
    echo "multi gpu training"
    export CUDA_VISIBLE_DEVICES=0,1,2,3     # which GPU to use
    python train.py \
        --use_cuda true
}

function run_train_multi_cpu() {
    echo "multi cpu training"
    python train.py \
        --use_cuda false \
        --cpu_num 10         #cpu_num works only when use_cuda=false
}

function run_eval() {
    echo "evaluating"
    echo "this may cost about 5 minutes if run on you CPU machine"
    python eval.py \
        --batch_size 200 \
        --word_emb_dim 128 \
        --grnn_hidden_dim 128 \
        --bigru_num 2 \
        --use_cuda False \
        --init_checkpoint ./model_baseline \
        --test_data ./data/test.tsv \
        --word_dict_path ./conf/word.dic \
        --label_dict_path ./conf/tag.dic \
        --word_rep_dict_path ./conf/q2b.dic
}


function run_infer() {
    echo "infering"
    python predict.py \
        --batch_size 200 \
        --word_emb_dim 128 \
        --grnn_hidden_dim 128 \
        --bigru_num 2 \
        --use_cuda False \
        --init_checkpoint ./model_baseline \
        --infer_data ./data/infer.tsv \
        --word_dict_path ./conf/word.dic \
        --label_dict_path ./conf/tag.dic \
        --word_rep_dict_path ./conf/q2b.dic
}


function run_inference() {
    echo "inference model"
    python inference_model.py \
        --word_emb_dim 128 \
        --grnn_hidden_dim 128 \
        --bigru_num 2 \
        --use_cuda False \
        --init_checkpoint ./model_baseline \
        --word_dict_path ./conf/word.dic \
        --label_dict_path ./conf/tag.dic \
        --word_rep_dict_path ./conf/q2b.dic \
        --inference_save_dir ./infer_model
}


function main() {
    local cmd=${1:-help}
    case "${cmd}" in
        train)
            run_train "$@";
            ;;
        train_single_gpu)
            run_train_single_gpu "$@";
            ;;
        train_multi_gpu)
            run_train_multi_gpu "$@";
            ;;
        train_multi_cpu)
            run_train_multi_cpu "$@";
            ;;
        eval)
            run_eval "$@";
            ;;
        infer)
            run_infer "$@";
            ;;
        inference)
            run_inference "$@";
            ;;
        help)
            echo "Usage: ${BASH_SOURCE} {train|train_single_gpu|train_multi_gpu|train_multi_cpu|eval|infer}";
            return 0;
            ;;
        *)
            echo "unsupport command [${cmd}]";
            echo "Usage: ${BASH_SOURCE} {train|train_single_gpu|train_multi_gpu|train_multi_cpu|eval|infer}";
            return 1;
            ;;
    esac
}

main "$@"
