#!/usr/bin/env bash
export FLAGS_enable_parallel_graph=1
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=3
export FLAGS_fraction_of_gpu_memory_to_use=0.95
TASK_NAME='simnet'
TRAIN_DATA_PATH=./data/train_pairwise_data
VALID_DATA_PATH=./data/test_pairwise_data
TEST_DATA_PATH=./data/test_pairwise_data
INFER_DATA_PATH=./data/infer_data
VOCAB_PATH=./data/term2id.dict
CKPT_PATH=./model_files
TEST_RESULT_PATH=./test_result
INFER_RESULT_PATH=./infer_result
TASK_MODE='pairwise'
CONFIG_PATH=./config/bow_pairwise.json
INIT_CHECKPOINT=./model_files/simnet_bow_pairwise_pretrained_model/


# run_train
train() {
    python run_classifier.py \
		--task_name ${TASK_NAME} \
		--use_cuda False \
		--do_train True \
		--do_valid True \
		--do_test True \
		--do_infer False \
		--batch_size 128 \
		--train_data_dir ${TRAIN_DATA_PATH} \
		--valid_data_dir ${VALID_DATA_PATH} \
		--test_data_dir ${TEST_DATA_PATH} \
		--infer_data_dir ${INFER_DATA_PATH} \
		--output_dir ${CKPT_PATH} \
		--config_path ${CONFIG_PATH} \
		--vocab_path ${VOCAB_PATH} \
		--epoch 40 \
		--save_steps 2000 \
		--validation_steps 200 \
		--compute_accuracy False \
		--lamda 0.958 \
		--task_mode ${TASK_MODE}\
		--init_checkpoint ""
}
#run_evaluate
evaluate() {
	python run_classifier.py \
		--task_name ${TASK_NAME} \
		--use_cuda false \
		--do_test True \
		--verbose_result True \
		--batch_size 128 \
		--test_data_dir ${TEST_DATA_PATH} \
		--test_result_path ${TEST_RESULT_PATH} \
		--config_path ${CONFIG_PATH} \
		--vocab_path ${VOCAB_PATH} \
		--task_mode ${TASK_MODE} \
		--compute_accuracy False \
		--lamda 0.958 \
		--init_checkpoint ${INIT_CHECKPOINT}
}
# run_infer
infer() {
	python run_classifier.py \
		--task_name ${TASK_NAME} \
		--use_cuda false \
		--do_infer True \
		--batch_size 128 \
		--infer_data_dir ${INFER_DATA_PATH} \
		--infer_result_path ${INFER_RESULT_PATH} \
		--config_path ${CONFIG_PATH} \
		--vocab_path ${VOCAB_PATH} \
		--task_mode ${TASK_MODE} \
		--init_checkpoint ${INIT_CHECKPOINT}
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
            echo "Unsupport commend [${cmd}]";
            echo "Usage: ${BASH_SOURCE} {train|eval|infer}";
            return 1;
            ;;
    esac
}
main "$@"
