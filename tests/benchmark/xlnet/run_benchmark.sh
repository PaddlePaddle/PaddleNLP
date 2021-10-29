#!/usr/bin/env bash
set -xe

# 运行示例：CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} 500 ${model_mode}
# 参数说明
function _set_params(){
    run_mode=${1:-"sp"}          # 单卡sp|多卡mp
    batch_size=${2:-"64"}
    fp_item=${3:-"fp32"}        # fp32|fp16
    max_iter=${4:-"500"}       # 可选，如果需要修改代码提前中断
    model_name=${5:-"model_name"}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}  # TRAIN_LOG_DIR 后续QA设置该参数

#   以下不用修改
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    log_file=${run_log_path}/${model_name}_${run_mode}_bs${batch_size}_${fp_item}_${num_gpu_devices}
}

python -m paddle.distributed.launch --gpus "0" ./run_glue.py \
    --model_name_or_path xlnet-base-cased \
    --task_name SST-2 \
    --max_seq_length 128 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --logging_steps 100 \
    --save_steps 500 \
    --output_dir ./tmp/


function _train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"

    train_cmd="--model_name_or_path=${model_name}
               --task_name=${task_name}
               --max_seq_length=${max_seq_length}
               --batch_size=${batch_size}
               --learning_rate=${learning_rate}
               --max_steps=${max_iter}
               --output_dir=${run_log_path}
               --fp=${fp_item}
               "

    case ${run_mode} in
    sp)
        train_cmd="python -m paddle.distributed.launch --gpus=$CUDA_VISIBLE_DEVICES ./run_glue.py ${train_cmd}" ;;
    mp)
        train_cmd="python -m paddle.distributed.launch --gpus=$CUDA_VISIBLE_DEVICES ./run_glue.py ${train_cmd}" ;;
        log_parse_file="mylog/workerlog.0" ;;
    *) echo "choose run_mode(sp or mp)"; exit 1;
    esac


# 以下不用修改
    timeout 15m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
        export job_fail_flag=1
    else
        echo -e "${model_name}, SUCCESS"
        export job_fail_flag=0
    fi
    kill -9 `ps -ef|grep 'python'|awk '{print $2}'`

    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}

_set_params $@
_train