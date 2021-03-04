#! /bin/bash
export FLAGS_enable_parallel_graph=1
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=0
export FLAGS_fraction_of_gpu_memory_to_use=0.95
export CPU_NUM=1

TASK_NAME='senta'
DATA_PATH=./senta_data/
CKPT_PATH=./save_models
MODEL_PATH=./save_models/step_1800/
MODEL_TYPE=${MODEL_TYPE:-bilstm_net}
USE_CUDA=${USE_CUDA:-true}

# run_train on train.tsv and do_val on test.tsv
train() {
    python -u run_classifier.py \
        --task_name ${TASK_NAME} \
        --use_cuda $USE_CUDA \
        --do_train true \
        --do_val false \
        --do_infer false \
        --model_type $MODEL_TYPE \
        --batch_size 8 \
        --data_dir ${DATA_PATH} \
        --vocab_path ${DATA_PATH}/word_dict.txt \
        --checkpoints ${CKPT_PATH} \
        --save_steps 50 \
        --validation_steps 50 \
        --epoch 3 \
        --senta_config_path ./senta_config.json \
        --skip_steps 10
}

# run_eval on test.tsv
evaluate() {
    python -u run_classifier.py \
        --task_name ${TASK_NAME} \
        --use_cuda $USE_CUDA \
        --do_train false \
        --do_val true \
        --do_infer false \
        --model_type $MODEL_TYPE \
        --batch_size 10 \
        --data_dir ${DATA_PATH} \
        --vocab_path ${DATA_PATH}/word_dict.txt \
        --init_checkpoint ${MODEL_PATH} \
        --senta_config_path ./senta_config.json 
}

# run_infer on infer.tsv
infer() {
    python -u run_classifier.py \
        --task_name ${TASK_NAME} \
        --use_cuda $USE_CUDA \
        --do_train false \
        --do_val false \
        --do_infer true \
        --model_type $MODEL_TYPE \
        --batch_size 10 \
        --data_dir ${DATA_PATH} \
        --vocab_path ${DATA_PATH}/word_dict.txt \
        --init_checkpoint ${MODEL_PATH} \
        --senta_config_path ./senta_config.json
}

# run_save_inference_model
save_inference_model() {
    python -u inference_model.py \
        --use_cuda false \
        --do_save_inference_model true \
        --init_checkpoint ${MODEL_PATH} \
        --inference_model_dir ./inference_model
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
