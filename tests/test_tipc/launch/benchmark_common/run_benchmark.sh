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
    model_item=${model_item:-"CE_autotuner_llama7b"}
    run_mode=${run_mode:-"pretrain"}
    device_num=${device_num:-"N1C8"}
    global_batch_size=${global_batch_size:-8}
    modle_json_file=${modle_json_file:-"./llama7b_pretrain_params.json"}
    base_batch_size=${global_batch_size}

    profiling=${PROFILING:-"false"}      # (必选) Profiling  开关，默认关闭，通过全局变量传递
    model_repo="PaddleNLP"          # (必选) 模型套件的名字
    speed_unit="tokens/s"         # (必选)速度指标单位
    skip_steps=0                  # (必选)解析日志，跳过模型前几个性能不稳定的step
    keyword="ips:"                 # (必选)解析日志，筛选出性能数据所在行的关键字
    convergence_key="loss:"        # (可选)解析日志，筛选出收敛数据所在行的关键字 如：convergence_key="loss:"

    fp_item="bf16"
    workerlog_id=0
    ip_lists=($(echo $TRAINER_INSTANCES))
    master_ip=${ip_lists[0]}
    nnodes=${nnodes:-1}
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
    is_large_model=True
}

function _train(){
    batch_size=${per_device_train_batch_size}  # 如果模型跑多卡单进程时,请在_train函数中计算出多卡需要的bs

    if [ -d $OUTPUT_PATH ]; then
        rm -rf $OUTPUT_PATH
    fi
    mkdir $OUTPUT_PATH

    echo "current CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, model_name=${model_name}, device_num=${device_num}, is profiling=${profiling}"

    if [ ${profiling} = "true" ];then
        add_options="--profiler_options=\"batch_range=[10,20];state=GPU;tracer_option=Default;profile_path=model.profile\""
        log_file=${profiling_log_file}
    else
        add_options=""
        log_file=${train_log_file}
    fi

    # 以下为通用执行命令，无特殊可不用修改
    case ${device_num} in
    N1C8) echo "Run with: device_num=${device_num}, run_mode=${run_mode}"
        train_cmd="python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 \
                   run_pretrain.py ${modle_json_file}"
        ;;
    case1_N2C16) echo "Run with: device_num=${device_num} run_mode=${run_mode}"
        unset PADDLE_ELASTIC_JOB_ID
        unset PADDLE_TRAINER_ENDPOINTS
        unset DISTRIBUTED_TRAINER_ENDPOINTS
        unset FLAGS_START_PORT
        unset PADDLE_ELASTIC_TIMEOUT
        unset PADDLE_TRAINERS_NUM
        unset PADDLE_TRAINER_ID
        train_cmd="python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 \
                   --master $master_ip:6768 --nnodes $nnodes \
                   run_pretrain.py ${modle_json_file}"
        ;;
    case2_N2C16) echo "Run with: device_num=${device_num} run_mode=${run_mode}"
        train_cmd="python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 \
                   --master $master_ip:6768 --nnodes $nnodes \
                   run_pretrain.py ${modle_json_file}"
        ;;
    case3_N2C16) echo "Run with: device_num=${device_num} run_mode=${run_mode}"
        export PADDLE_MASTER=$master_ip:6768
        export PADDLE_NNODES=$nnodes        
        train_cmd="python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 \
                   run_pretrain.py ${modle_json_file}"
        ;;
    case4_N2C16) echo "Run with: device_num=${device_num} run_mode=${run_mode}"
        train_cmd="python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 \
                   --ips $ip_lists \
                   run_pretrain.py ${modle_json_file}" 
        ;;
    *) echo "Run with: device_num=${device_num}, run_mode=${run_mode}"
        train_cmd="python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 \
            run_pretrain.py ${modle_json_file}"
        ;;
    esac
    cd ../llm/llama
    echo "train_cmd: ${train_cmd}  log_file: ${log_file}"
    python -c "import paddlenlp"
    echo $PADDLE_ELASTIC_JOB_ID 
    echo $PADDLE_TRAINER_ENDPOINTS 
    echo $DISTRIBUTED_TRAINER_ENDPOINTS 
    echo $FLAGS_START_PORT 
    echo $PADDLE_ELASTIC_TIMEOUT 
    echo $PADDLE_TRAINERS_NUM 
    echo $PADDLE_TRAINER_ID 
    timeout 10m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
    else
        echo -e "${model_name}, SUCCESS"
    fi
}

export PYTHONPATH=$(dirname "$PWD"):$PYTHONPATH
source ${BENCHMARK_ROOT}/scripts/run_model.sh   # 在该脚本中会对符合benchmark规范的log使用analysis.py 脚本进行性能数据解析;如果不联调只想要产出训练log可以注掉本行,提交时需打开
_set_params $@
#_train       # 如果只产出训练log,不解析,可取消注释
_run     # 该函数在run_model.sh中,执行时会调用_train; 如果不联调只产出训练log可以注掉本行,提交时需打开