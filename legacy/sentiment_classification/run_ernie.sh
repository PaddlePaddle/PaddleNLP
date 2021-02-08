#! /bin/bash
export FLAGS_fraction_of_gpu_memory_to_use=0.95
export FLAGS_enable_parallel_graph=1
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=0
export CPU_NUM=1
ERNIE_PRETRAIN=./ernie_pretrain_model/
DATA_PATH=./senta_data
MODEL_SAVE_PATH=./save_models

# run_train
train() {
    python -u run_ernie_classifier.py \
        --use_cuda true \
        --verbose true \
        --do_train true \
        --do_val true \
        --do_infer false \
        --use_paddle_hub false \
        --batch_size 4 \
        --init_checkpoint $ERNIE_PRETRAIN/params \
        --train_set $DATA_PATH/train.tsv \
        --dev_set $DATA_PATH/dev.tsv \
        --test_set $DATA_PATH/test.tsv \
        --vocab_path $ERNIE_PRETRAIN/vocab.txt \
        --checkpoints $MODEL_SAVE_PATH \
        --save_steps 5000 \
        --validation_steps 5000 \
        --epoch 2 \
        --max_seq_len 256 \
        --ernie_config_path $ERNIE_PRETRAIN/ernie_config.json \
        --model_type "ernie_base" \
        --lr 5e-5 \
        --skip_steps 10 \
        --num_labels 2 \
        --random_seed 1
}

# run_eval
evaluate() {
    python -u run_ernie_classifier.py \
        --use_cuda true \
        --verbose true \
        --do_train false \
        --do_val true \
        --do_infer false \
        --use_paddle_hub false \
        --batch_size 4 \
        --init_checkpoint ./save_models/step_4801/ \
        --dev_set $DATA_PATH/dev.tsv \
        --vocab_path $ERNIE_PRETRAIN/vocab.txt \
        --max_seq_len 256 \
        --ernie_config_path $ERNIE_PRETRAIN/ernie_config.json \
        --model_type "ernie_base" \
        --num_labels 2
    
    python -u run_ernie_classifier.py \
        --use_cuda true \
        --verbose true \
        --do_train false \
        --do_val true \
        --do_infer false \
        --use_paddle_hub false \
        --batch_size 4 \
        --init_checkpoint ./save_models/step_4801/ \
        --dev_set $DATA_PATH/test.tsv \
        --vocab_path $ERNIE_PRETRAIN/vocab.txt \
        --max_seq_len 256 \
        --ernie_config_path $ERNIE_PRETRAIN/ernie_config.json \
        --model_type "ernie_base" \
        --num_labels 2
}

# run_infer
infer() {
    python -u run_ernie_classifier.py \
        --use_cuda true \
        --verbose true \
        --do_train false \
        --do_val false \
        --do_infer true \
        --use_paddle_hub false \
        --batch_size 4 \
        --init_checkpoint ./save_models/step_4801 \
        --test_set $DATA_PATH/test.tsv \
        --vocab_path $ERNIE_PRETRAIN/vocab.txt \
        --max_seq_len 256 \
        --ernie_config_path $ERNIE_PRETRAIN/ernie_config.json \
        --model_type "ernie_base" \
        --num_labels 2
}

# run_save_inference_model
save_inference_model() {
    python -u inference_model_ernie.py \
        --use_cuda true \
        --do_save_inference_model true \
        --init_checkpoint ./save_models/step_4801/ \
        --inference_model_dir ./inference_model \
        --ernie_config_path $ERNIE_PRETRAIN/ernie_config.json \
        --model_type "ernie_base" \
        --vocab_path $ERNIE_PRETRAIN/vocab.txt \
        --test_set ${DATA_PATH}/test.tsv    \
        --batch_size 4
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
        save_inference_model)
            save_inference_model "$@";
            ;;
        help)
            echo "Usage: ${BASH_SOURCE} {train|eval|infer|save_inference_model}";
            return 0;
            ;;
        *)
            echo "Unsupport commend [${cmd}]";
            echo "Usage: ${BASH_SOURCE} {train|eval|infer|save_inference_model}";
            return 1;
            ;;
    esac
}
main "$@"
