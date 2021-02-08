#set -eux
export FLAGS_fraction_of_gpu_memory_to_use=0.02
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fast_eager_deletion_mode=1
# export FLAGS_sync_nccl_allreduce=1
# export NCCL_DEBUG=INFO
# export NCCL_IB_GID_INDEX=3
# export GLOG_v=1
# export GLOG_logtostderr=1
export CUDA_VISIBLE_DEVICES=0        # which GPU to use

ERNIE_PRETRAINED_MODEL_PATH=./pretrained/
ERNIE_FINETUNED_MODEL_PATH=./model_finetuned
DATA_PATH=./data/
# train
function run_train() {
    echo "training"
    python run_ernie_sequence_labeling.py \
        --mode train \
        --ernie_config_path "${ERNIE_PRETRAINED_MODEL_PATH}/ernie_config.json" \
        --model_save_dir "./ernie_models" \
        --init_pretraining_params "${ERNIE_PRETRAINED_MODEL_PATH}/params/" \
        --epoch 10 \
        --save_steps 5 \
        --validation_steps 5 \
        --base_learning_rate 2e-4 \
        --crf_learning_rate 0.2 \
        --init_bound 0.1 \
        --print_steps 1 \
        --vocab_path "${ERNIE_PRETRAINED_MODEL_PATH}/vocab.txt" \
        --batch_size 3 \
        --random_seed 0 \
        --num_labels 57 \
        --max_seq_len 128 \
        --train_data "${DATA_PATH}/train.tsv" \
        --test_data "${DATA_PATH}/test.tsv" \
        --label_map_config "./conf/label_map.json" \
        --do_lower_case true \
        --use_cuda false \
        --cpu_num 1
}

function run_train_single_gpu() {
    echo "single gpu training"              # which GPU to use
    export CUDA_VISIBLE_DEVICES=0
    python run_ernie_sequence_labeling.py \
        --mode train \
        --ernie_config_path "${ERNIE_PRETRAINED_MODEL_PATH}/ernie_config.json" \
        --init_pretraining_params "${ERNIE_PRETRAINED_MODEL_PATH}/params/" \
        --vocab_path "${ERNIE_PRETRAINED_MODEL_PATH}/vocab.txt" \
        --use_cuda true
}


function run_train_multi_cpu() {
    echo "multi cpu training"
    python run_ernie_sequence_labeling.py \
        --mode train \
        --ernie_config_path "${ERNIE_PRETRAINED_MODEL_PATH}/ernie_config.json" \
        --init_pretraining_params "${ERNIE_PRETRAINED_MODEL_PATH}/params/" \
        --vocab_path "${ERNIE_PRETRAINED_MODEL_PATH}/vocab.txt" \
        --use_cuda false \
        --batch_size 64 \
        --cpu_num 8         #cpu_num works only when use_cuda=false
}


function run_eval() {
    echo "evaluating"
    python run_ernie_sequence_labeling.py \
        --mode eval \
        --ernie_config_path "${ERNIE_PRETRAINED_MODEL_PATH}/ernie_config.json" \
        --init_checkpoint "${ERNIE_FINETUNED_MODEL_PATH}" \
        --init_bound 0.1 \
        --vocab_path "${ERNIE_PRETRAINED_MODEL_PATH}/vocab.txt" \
        --batch_size 64 \
        --random_seed 0 \
        --num_labels 57 \
        --max_seq_len 128 \
        --test_data "${DATA_PATH}/test.tsv" \
        --label_map_config "./conf/label_map.json" \
        --do_lower_case true \
        --use_cuda false

}

function run_infer() {
    echo "infering"
    python run_ernie_sequence_labeling.py \
        --mode infer \
        --ernie_config_path "${ERNIE_PRETRAINED_MODEL_PATH}/ernie_config.json" \
        --init_checkpoint "${ERNIE_FINETUNED_MODEL_PATH}" \
        --init_bound 0.1 \
        --vocab_path "${ERNIE_PRETRAINED_MODEL_PATH}/vocab.txt" \
        --batch_size 64 \
        --random_seed 0 \
        --num_labels 57 \
        --max_seq_len 128 \
        --test_data "${DATA_PATH}/test.tsv" \
        --label_map_config "./conf/label_map.json" \
        --do_lower_case true \
        --use_cuda false

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
        train_multi_cpu)
            run_train_multi_cpu "$@";
            ;;
        eval)
            run_eval "$@";
            ;;
        infer)
            run_infer "$@";
            ;;
        help)
            echo "Usage: ${BASH_SOURCE} {train|train_single_gpu|train_multi_cpu|eval|infer}";
            return 0;
            ;;
        *)
            echo "unsupport command [${cmd}]";
            echo "Usage: ${BASH_SOURCE} {train|train_single_gpu|train_multi_cpu|eval|infer}";
            return 1;
            ;;
    esac
}

main "$@"
