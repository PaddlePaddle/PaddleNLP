export PATH="/home/guohongjie/tmp/paddle/paddle_release_home/python/bin/:$PATH"





#  CPU setting
:<<EOF
USE_CUDA=false
CPU_NUM=3 # cpu_num works only when USE_CUDA=false
# path to your python
export PATH="/home/work/guohongjie/cpu_paddle/python2/bin:$PATH"
EOF



# GPU_settting
:<<EOF
# cuda path
LD_LIBRARY_PATH=/home/work/cuda/cudnn/cudnn_v7/cuda/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH="/home/work/guohongjie/cuda/cudnn/cudnn_v7/cuda/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/work/guohongjie/cuda/cuda-9.0/lib64:$LD_LIBRARY_PATH"
USE_CUDA=true
CPU_NUM=3 # cpu_num works only when USE_CUDA=false
export FLAGS_fraction_of_gpu_memory_to_use=0.02
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fast_eager_deletion_mode=1
export CUDA_VISIBLE_DEVICES=0     #   which GPU to use
# path to your python
export PATH="/home/work/guohongjie/gpu_paddle/python2/bin:$PATH"
EOF



echo "the python your use is `which python`"

MODEL_PATH=None # not loading any pretrained model
#MODEL_PATH=./model/ # the default pretrained model
INPUT_DIR=./data/input/
OUTPUT_DIR=./data/output/
TRAIN_CONF=./data/input/model.conf
BUILD_DICT=false	# if you use your new dataset, set it true to build domain and char dict
BATCH_SIZE=64




train() {
      python -u run_classifier.py \
        --use_cuda ${USE_CUDA} \
        --cpu_num ${CPU_NUM} \
        --do_train true \
        --do_eval false \
        --do_test false \
        --build_dict ${BUILD_DICT} \
        --data_dir ${INPUT_DIR} \
        --save_dir ${OUTPUT_DIR} \
        --config_path ${TRAIN_CONF} \
        --batch_size ${BATCH_SIZE} \
        --init_checkpoint ${MODEL_PATH} 
}

evaluate() {
    python -u run_classifier.py \
        --use_cuda ${USE_CUDA} \
        --cpu_num ${CPU_NUM} \
        --do_train true \
        --do_eval true \
        --do_test false \
        --build_dict ${BUILD_DICT} \
        --data_dir ${INPUT_DIR} \
        --save_dir ${OUTPUT_DIR} \
        --config_path ${TRAIN_CONF} \
        --batch_size ${BATCH_SIZE}  \
        --init_checkpoint ${MODEL_PATH} 
}


infer() {
    python -u run_classifier.py \
        --use_cuda ${USE_CUDA} \
        --cpu_num ${CPU_NUM} \
        --do_train false \
        --do_eval false \
        --do_test true \
        --build_dict ${BUILD_DICT} \
        --data_dir ${INPUT_DIR} \
        --save_dir ${OUTPUT_DIR} \
        --config_path ${TRAIN_CONF} \
        --batch_size ${BATCH_SIZE}  \
        --init_checkpoint ${MODEL_PATH} 
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
        test)
            infer "$@";
            ;;
        help)
            echo "Usage: ${BASH_SOURCE} {train|eval|test}";
            return 0;
            ;;
        *)
            echo "Unsupport commend [${cmd}]";
            echo "Usage: ${BASH_SOURCE} {train|eval|test}";
            return 1;
            ;;
    esac
}

main "$@"
