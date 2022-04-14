#!/usr/bin/env bash
set -xe
# Test training benchmark for a model.
# Usage：CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} ${max_iter} ${model_item}

function _set_params(){
    run_mode=${1:-"sp"}         # sp or mp
    batch_size=${2:-"2"}
    fp_item=${3:-"fp32"}        # fp32 or fp16
    max_iter=${4:-"100"}
    model_item=${5:-"gpt2"}
    mode_item=${6:-"static"}
    need_profile=${7:-"off"}
    
    mission_name="语义表示"
    direction_id=1
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}

    base_batch_size=$(($batch_size*1024))
    model_name=${model_item}_${mode_item}_bs${batch_size}_${fp_item}
    log_file=${run_log_path}/${model_name}_${num_gpu_devices}_${run_mode}
    log_folder=${run_log_path}/${model_item}_logdir
    log_profile=${run_log_path}/${model_item}_model.profile
    OUTPUT_PATH=${run_log_path}/output


    log_with_profiler=$log_file
    profiler_path=$log_profile
    keyword="ips:" 
    keyword_loss="loss:"
    skip_steps=20
    model_mode=-1
    ips_unit='tokens/s'
    index="1"
    gpu_num=$num_gpu_devices 
}


function _train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"

    if [ -d $OUTPUT_PATH ]; then
        rm -rf $OUTPUT_PATH
    fi

    if [ $fp_item = "fp16" ]; then
        use_fp16_cmd="--use_amp true" 
        if [ $mode_item = "dygraph" ] && [ $model_item = "gpt3" ]; then
            use_fp16_cmd="--use_pure_fp16 true"
        fi
    fi

    profiler_cmd=""
    profiler_options="batch_range=[100,110];profile_path=${log_profile}"
    if [ $need_profile = "on" ]; then
        profiler_cmd="--profiler_options=${profiler_options}"
    fi

    script_cmd="run_pretrain_static.py"
    if [ $mode_item = "dygraph" ]; then
        script_cmd="run_pretrain.py"
    fi

    base_path="examples/language_model/gpt/"
    if [ $model_item = 'gpt3' ]; then
        base_path=examples/language_model/gpt-3/${mode_item}
    fi
    data_path=$(pwd)"/data"

    train_cmd="${profiler_cmd}\
               --micro_batch_size=${batch_size} \
               --global_batch_size=$((${batch_size}*${num_gpu_devices})) \
               --model_type="gpt"\
               --model_name_or_path="gpt2-en"\
               --input_dir=${data_path}\
               --output_dir=${OUTPUT_PATH} \
               --dp_degree=${num_gpu_devices}\
               --max_seq_len 1024 \
               --max_lr 0.00015 \
               --min_lr 0.00001 \
               --max_steps=${max_iter} \
               --save_steps 100000 \
               --decay_steps 320000 \
               --weight_decay 0.01\
               --warmup_rate 0.01 \
               --grad_clip 1.0 \
               --logging_freq 1\
               --eval_freq 1000 \
               --device "gpu" \
               ${use_fp16_cmd}"

    case ${run_mode} in
    sp)
        train_cmd="python -m paddle.distributed.launch --log_dir=${log_folder} --gpus=$CUDA_VISIBLE_DEVICES \
                  ${script_cmd} ${train_cmd}" ;;
    mp)
        train_cmd="python -m paddle.distributed.launch --log_dir=${log_folder} --gpus=$CUDA_VISIBLE_DEVICES \
                  ${script_cmd} ${train_cmd}" ;;
    *) echo "choose run_mode(sp or mp)"; exit 1;
    esac

    #timeout 1s 
    #eval` $train_cmd
    cd ${base_path}
    timeout 15m  ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
        export job_fail_flag=1
    else
        echo -e "${model_name}, SUCCESS"
        export job_fail_flag=0
    fi
    cd -
    #kill -9 `ps -ef|grep 'python'|awk '{print $2}'`

    rm ${log_file}
    cp ${log_folder}/workerlog.0 ${log_file}
    rm -r ${log_folder}
}

source ${BENCHMARK_ROOT}/scripts/run_model.sh
_set_params $@
_run
