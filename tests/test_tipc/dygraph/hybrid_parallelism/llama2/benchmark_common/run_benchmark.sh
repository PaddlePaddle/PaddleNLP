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

    is_large_model=True
    base_batch_size=${global_batch_size}
    profiling=${PROFILING:-"false"}      # (必选) Profiling  开关，默认关闭，通过全局变量传递
    model_repo="PaddleNLP"          # (必选) 模型套件的名字
    speed_unit="tokens/s"         # (必选)速度指标单位
    skip_steps=0                  # (必选)解析日志，跳过模型前几个性能不稳定的step
    keyword="ips:"                 # (必选)解析日志，筛选出性能数据所在行的关键字
    convergence_key="loss:"        # (可选)解析日志，筛选出收敛数据所在行的关键字 如：convergence_key="loss:"
    is_large_model=True            # (可选) True为大模型，获取最后一行ips数据，不计算均值

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
    cd ../llm/llama
    rm -rf ./auto_config_${MODEL_TYPE}/*GBS*
    rm -rf ./auto_config_${MODEL_TYPE}/*auto_tuner.log
    rm -rf ./auto_config_${MODEL_TYPE}/*csv
    rm -rf ./auto_config_${MODEL_TYPE}/best_*
    rm -rf mylog && rm -rf checkpoints
    
    echo "train_cmd: ${train_cmd}  log_file: ${log_file}"
    timeout 15m ${train_cmd} > ${log_file} 2>&1

    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
    else
        echo -e "${model_name}, SUCCESS"
    fi
    # 计算方式更新并重写
    interval_samples_per_second=`cat ${log_file} | grep 'global_step: 50' \
        | awk -F 'interval_samples_per_second: ' '{print $2}'  | awk -F ',' '{print $1}' | tail -n 1`
    seq_length=4096
    num_total_cards=$(echo $nnodes $TRAINER_GPU_CARD_COUNT |awk '{printf "%d\n", $1*$2}')
    ips_per_card=$(echo $interval_samples_per_second $seq_length $num_total_cards|awk '{printf "%0.2f\n", $1*$2/$3}')
    echo "ips: $ips_per_card tokens/s" >> ${log_file}

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
