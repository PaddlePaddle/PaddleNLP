#!/bin/bash
export FLAGS_enable_parallel_graph=1
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=0
export FLAGS_fraction_of_gpu_memory_to_use=0.95
CKPT_PATH=./save_models/textcnn
MODEL_PATH=./save_models/textcnn/step_756

# run_train on train.tsv and do_val on dev.tsv
train() {
    python run_classifier.py \
        --use_cuda false \
        --do_train true \
        --do_val true \
        --epoch 5 \
        --lr 0.002 \
        --batch_size 64 \
        --save_checkpoint_dir ${CKPT_PATH} \
        --save_steps 200 \
        --validation_steps 200 \
        --skip_steps 200
}
# run_eval on test.tsv
evaluate() {
    python run_classifier.py \
        --use_cuda false \
        --do_val true \
        --batch_size 128 \
        --init_checkpoint ${MODEL_PATH}
}
# run_infer on infer.tsv
infer() {
    python run_classifier.py \
        --use_cuda false \
        --do_infer true \
        --batch_size 32 \
        --init_checkpoint ${MODEL_PATH}
}

# run_save_inference_model
save_inference_model() {
    python inference_model.py \
        --use_cuda false \
        --do_save_inference_model true \
        --init_checkpoint  ${MODEL_PATH} \
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
            echo "unsupport command [${cmd}]";
            echo "Usage: ${BASH_SOURCE} {train|eval|infer|save_inference_model}";
            return 1;
            ;;
    esac
}
main "$@"
