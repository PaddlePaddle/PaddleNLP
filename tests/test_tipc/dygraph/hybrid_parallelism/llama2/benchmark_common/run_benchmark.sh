#!/usr/bin/env bash

# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Test training benchmark for a model.
# Usage：bash benchmark/run_benchmark.sh ${model_name_or_path} ${per_device_train_batch_size} ${tensor_parallel_degree} ${pipeline_parallel_degree} ${virtual_pp_degree} ${sequence_parallel} ${sharding_parallel_degree} ${sharding} ${recompute} ${run_mode} ${device_num}
function _set_params(){
    model_item=${model_item:-"meta-llama-Llama-2-7b_pretrain"}
    run_mode=${run_mode:-"MP2-PP1"}
    device_num=${device_num:-"N1C8"}
    global_batch_size=${global_batch_size:-64}
    fp_item="bf16"
    MODEL_TYPE=${model_type:-"llama2_7b"}

    ip_lists=($(echo $TRAINER_INSTANCES | tr ',' ' '))
    master_ip=${ip_lists[0]}
    nnodes=${nnodes:-1}

    base_batch_size=${global_batch_size}
    profiling=${PROFILING:-"false"}      # (必选) Profiling  开关，默认关闭，通过全局变量传递
    model_repo="PaddleNLP"          # (必选) 模型套件的名字
    speed_unit="tokens/s"         # (必选)速度指标单位
    skip_steps=10                  # (必选)解析日志，跳过模型前几个性能不稳定的step
    keyword="interval_tokens_per_second_per_device:"                 # (必选)解析日志，筛选出性能数据所在行的关键字
    convergence_key="loss:"        # (可选)解析日志，筛选出收敛数据所在行的关键字 如：convergence_key="loss:"
    model_mode=5                   # 获取ips数据及单位，仅跳过skip_steps后计算均值，单位保持token/s不变
    
    # 以下为通用执行命令，无特殊可不用修改
    model_name=${model_item}_bs${global_batch_size}_${fp_item}_${run_mode}  # (必填) 且格式不要改动,与竞品名称对齐
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}  # （必填） TRAIN_LOG_DIR  benchmark框架设置该参数为全局变量
    profiling_log_path=${PROFILING_LOG_DIR:-$(pwd)}  # （必填） PROFILING_LOG_DIR benchmark框架设置该参数为全局变量
    speed_log_path=${LOG_PATH_INDEX_DIR:-$(pwd)}
    train_log_file=${run_log_path}/${model_repo}_${model_name}_${device_num}_log
    mkdir -p $(dirname ${train_log_file})

    profiling_log_file=${profiling_log_path}/${model_repo}_${model_name}_${device_num}_profiling
    mkdir -p $(dirname ${profiling_log_file})

    speed_log_file=${speed_log_path}/${model_repo}_${model_name}_${device_num}_speed
    mkdir -p $(dirname ${speed_log_file})

    OUTPUT_PATH=${run_log_path}/output
}

# 循环监控文件写入状态和进程状态
monitor_log_file() {
    local log_file="$1"  # 获取日志文件路径
    local training_pid="$2"  # 获取训练进程的 PID
    local no_update_duration=0  # 初始化无更新时长计数
    local last_size=0
    local kill_flag_file="/tmp/monitor_killed_$training_pid"

    echo "$(date '+%Y-%m-%d %H:%M:%S') 开始监控进程 $training_pid 和日志文件 $log_file..."

    while true; do
        sleep 5  # 每隔 5 秒检查一次日志文件

        # 判断日志文件是否存在
        if [ ! -f "$log_file" ]; then
            echo "日志文件 $log_file 不存在，检查进程状态..."
            # 如果日志文件不存在，直接判断进程是否结束
            if ! ps -p $training_pid > /dev/null; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') 进程 $training_pid 已经结束。"
                break
            fi
            continue  # 如果文件不存在，跳过后续逻辑，继续循环
        fi

        # 获取当前日志文件的大小
        new_size=$(stat -c %s "$log_file")

        if [ "$last_size" -eq "$new_size" ]; then
            # 文件大小未变化，增加无更新时长计数
            no_update_duration=$((no_update_duration + 5))
            echo "$(date '+%Y-%m-%d %H:%M:%S') 文件未写入..."
            if [ "$no_update_duration" -ge 180 ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') 文件在过去的 3 分钟内没有继续写入，准备杀掉进程 $training_pid."
                # 创建标志文件
                touch "$kill_flag_file"
                ls -l "$kill_flag_file"
                kill -9 $training_pid  # 杀掉进程
                echo "$(date '+%Y-%m-%d %H:%M:%S') 进程 $training_pid 已经被杀掉。"
                break
            fi
        else
            # 文件大小有变化，重置无更新时长计数
            echo "$(date '+%Y-%m-%d %H:%M:%S') 文件仍在写入..."
            no_update_duration=0
            last_size=$new_size
        fi

        # 如果训练进程已经结束，退出监控
        if ! ps -p $training_pid > /dev/null; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') 进程 $training_pid 已经结束。"
            break
        fi
    done
}

function _train(){
    batch_size=${per_device_train_batch_size}  # 如果模型跑多卡单进程时,请在_train函数中计算出多卡需要的bs

    if [ -d $OUTPUT_PATH ]; then
        rm -rf $OUTPUT_PATH
    fi
    mkdir $OUTPUT_PATH

    echo "current CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, model_name=${model_name}, device_num=${device_num}, is profiling=${profiling}"

    if [ ${profiling} == "true" ];then
        add_options="--profiler_options=\"batch_range=[10,20];state=GPU;tracer_option=Default;profile_path=model.profile\""
        log_file=${profiling_log_file}
    else
        add_options=""
        log_file=${train_log_file}
    fi

    # Disable for hanging bug
    # if [ "${tensor_parallel_degree}" != "1" ]; then
    #     export CUDA_DEVICE_MAX_CONNECTIONS=1
    # fi

    if [ ${run_mode} == "autotuner" ]; then
        unset PADDLE_ELASTIC_JOB_ID
        unset PADDLE_TRAINER_ENDPOINTS
        unset DISTRIBUTED_TRAINER_ENDPOINTS
        unset FLAGS_START_PORT
        unset PADDLE_ELASTIC_TIMEOUT
        unset PADDLE_TRAINERS_NUM
        unset PADDLE_TRAINER_ID
        autoconfig_args="--auto_tuner_json ./auto_config_${MODEL_TYPE}/${MODEL_TYPE}_pretrain_autoconfig.json"
    else
        autoconfig_args=""
    fi
    
    if [ ${PADDLE_TRAINER_ID} ]; then
        PADDLE_RANK_OPTION=" --rank ${PADDLE_TRAINER_ID}"
    else
        PADDLE_RANK_OPTION=""
    fi

    if [ "$autoconfig_args" != "" ]; then
        distributed_args="--master etcd://$master_ip:2379 --nnodes $nnodes:$nnodes"
    else
        distributed_args="--master $master_ip:36677 --nnodes $nnodes ${PADDLE_RANK_OPTION} --run_mode=collective"
    fi

    echo "==========System Env============="
    env
    echo "================================="

    # 以下为通用执行命令，无特殊可不用修改
    case ${device_num} in
    N1C8) echo "Run with: device_num=${device_num}, run_mode=${run_mode}"
        train_cmd="python -u -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 \
            --nnodes 1 --nproc_per_node 8 \
            ${autoconfig_args} --log_dir mylog run_pretrain.py \
            ./auto_config_${MODEL_TYPE}/pretrain-${MODEL_TYPE}-auto_tuner.json"
        ;;
    N4C32) echo "Run with: device_num=${device_num} run_mode=${run_mode}"
        train_cmd="python -u -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 \
            ${distributed_args} ${autoconfig_args} --log_dir mylog run_pretrain.py \
            ./auto_config_${MODEL_TYPE}/pretrain-${MODEL_TYPE}-auto_tuner.json"
        ;;
    *) echo "Run with: device_num=${device_num}, run_mode=${run_mode}"
        train_cmd="python -u -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 \
             ${distributed_args} ${autoconfig_args} --log_dir mylog run_pretrain.py \
            ./auto_config_${MODEL_TYPE}/pretrain-${MODEL_TYPE}-auto_tuner.json"
        ;;
    esac
    cd ../llm
    rm -rf ./auto_config_${MODEL_TYPE}/*GBS*
    rm -rf ./auto_config_${MODEL_TYPE}/*auto_tuner.log
    rm -rf ./auto_config_${MODEL_TYPE}/*csv
    rm -rf ./auto_config_${MODEL_TYPE}/best_*
    rm -rf mylog && rm -rf checkpoints

    echo "train_cmd: ${train_cmd}  log_file: ${log_file}"
    timeout 40m ${train_cmd} > ${log_file} 2>&1 &
    training_pid=$!  # 获取后台进程的 PID

    # 监控进程和日志的更新状态
    monitor_log_file "$log_file" "$training_pid" & 
    monitor_log_file_pid=$!  # 获取日志监控进程的 PID

    # 等待训练进程完成
    wait $training_pid
    exit_code=$?

    # 获取训练进程的退出码
    echo "训练进程 $training_pid 的退出码是 $exit_code"

    # 清理后台日志监控进程
    kill $monitor_log_file_pid


    if [ ${exit_code} -ne 0 ];then
        echo -e "${model_name}, FAIL"
        # 如果程序是主动报错退出，不是monitor_log_file函数kill掉的情况下，需要等待其它机器被kill
        # 标志文件位置
        kill_flag_file="/tmp/monitor_killed_$training_pid"
        if [ -f "$kill_flag_file" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') 训练进程 $training_pid 是被 monitor_log_file 函数杀掉的。"
            rm -f "$kill_flag_file"  # 清理标志文件
        else
            echo "$(date '+%Y-%m-%d %H:%M:%S') 训练进程 $training_pid 是主动报错退出的。"
            sleep 120
        fi
    else
        echo -e "${model_name}, SUCCESS"
    fi


    #kill -9 `ps -ef|grep 'python'|awk '{print $2}'`
    if [ ${device_num} != "N1C1" -a -d ./auto_config_${MODEL_TYPE}/best_cfg ]; then
        case_path=$PWD && cd - && mkdir -p mylog      # PaddleNLP/tests/mylog
        cp -r ${case_path}/autoconfig/best_cfg/workerlog.* ./mylog/
        cp -r ${case_path}/autoconfig/*.csv $(dirname "$log_file")
    else
        case_path=$PWD && cd - && mkdir -p mylog      # PaddleNLP/tests/mylog
        cp -r ${case_path}/mylog/workerlog.* ./mylog/
    fi
}

export FLAGS_selected_gpus="0,1,2,3,4,5,6,7"
export NCCL_IB_DISABLE=0
export PYTHONPATH=$(dirname "$PWD"):$PYTHONPATH

source ${BENCHMARK_ROOT}/scripts/run_model.sh   # 在该脚本中会对符合benchmark规范的log使用analysis.py 脚本进行性能数据解析;如果不联调只想要产出训练log可以注掉本行,提交时需打开
_set_params $@
#_train       # 如果只产出训练log,不解析,可取消注释
_run     # 该函数在run_model.sh中,执行时会调用_train; 如果不联调只产出训练log可以注掉本行,提交时需打开
