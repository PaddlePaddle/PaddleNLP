#!/usr/bin/env bash

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
    model_name_or_path=${model_name_or_path:-"Qwen/Qwen2.5-7B"}
    per_device_train_batch_size=${per_device_train_batch_size:-1}
    tensor_parallel_degree=${tensor_parallel_degree:-1}
    pipeline_parallel_degree=${pipeline_parallel_degree:-1}
    sharding_parallel_degree=${sharding_parallel_degree:-1}
    gradient_accumulation_steps=${gradient_accumulation_steps:-1}
    run_stage=${run_stage:-"sft"}
    run_mode=${run_mode:-"DP1-MP1-PP4-mbs1-acc8-recompute"}
    device_num=${device_num:-"N1C8"}
    global_batch_size=${global_batch_size:-16}
    model_item=${model_item:-"qwen-qwen-7b_seqlen2048_pretrain"}
    base_batch_size=${global_batch_size}

    profiling=${PROFILING:-"false"}      # (必选) Profiling  开关，默认关闭，通过全局变量传递
    model_repo="PaddleNLP"          # (必选) 模型套件的名字
    speed_unit="tokens/s"         # (必选)速度指标单位
    skip_steps=0                  # (必选)解析日志，跳过模型前几个性能不稳定的step
    keyword="Effective_Tokens_per_second_per_gpu:"                 # (必选)解析日志，筛选出性能数据所在行的关键字
    is_large_model=True           # (可选)普通模型默认为False，如果添加大模型且只取一条ips设置为True
    convergence_key="Total_Tokens_per_second_per_gpu:"        # (可选)解析日志，筛选出收敛数据所在行的关键字 如：convergence_key="loss:"

    fp_item="bf16"
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

    if [ ${profiling} = "true" ];then
        add_options="--profiler_options=\"batch_range=[10,20];state=GPU;tracer_option=Default;profile_path=model.profile\""
        log_file=${profiling_log_file}
    else
        add_options=""
        log_file=${train_log_file}
    fi

    case ${run_stage} in
    sft) train_cmd="run_finetune.py benchmark_json/${model_item%_*}/sft.json" ;;
    lora) train_cmd="run_finetune.py benchmark_json/${model_item%_*}/lora.json" ;;
    dpo) train_cmd="alignment/dpo/run_dpo.py benchmark_json/${model_item%_*}/dpo.json" ;;
    *) echo "choose run_stage(sft | lora | dpo)"; exit 1;
    esac

    if [ ${PADDLE_TRAINER_ID} ]
    then
        PADDLE_RANK_OPTION=" --rank ${PADDLE_TRAINER_ID}"
    else
        PADDLE_RANK_OPTION=""
    fi
    # 以下为通用执行命令，无特殊可不用修改
    case ${device_num} in
    N1C1) echo "Run with: device_num=${device_num}, run_mode=${run_mode}, run_stage=${run_stage}"
        train_cmd="python -u -m paddle.distributed.launch --log_dir=./mylog --gpus=0 ${PADDLE_RANK_OPTION}\
            ${train_cmd}"
        workerlog_id=0
        ;;
    *) echo "Run with: device_num=${device_num}, run_mode=${run_mode}, run_stage=${run_stage}"
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --gpus=0,1,2,3,4,5,6,7 ${PADDLE_RANK_OPTION}\
            ${train_cmd}"
        workerlog_id=0
        ;;
    esac
    cd ../llm/
    export no_proxy=bcebos.com
    echo "train_cmd: ${train_cmd}  log_file: ${log_file}"
    python -c "import paddlenlp"
    if [[ ${model_name_or_path} =~ "CE" ]];then # CE精度-不限制执行时间
        ${train_cmd} > ${log_file} 2>&1
    else
        timeout 60m ${train_cmd} > ${log_file} 2>&1
        # echo ${train_cmd}
        Effective_Tokens_per_second=`cat ${log_file} | grep -E 'Effective_Tokens_per_second|Effective tokens per second:' \
                                            |awk -F': ' '{print $2}' |awk -F' ' '{print $1}'`
        num_gpu=$(echo "$device_num" | sed 's/^.*C//')
        Effective_Tokens_per_second_per_gpu=$(awk -v a="$Effective_Tokens_per_second" -v b="$num_gpu" 'BEGIN {printf "%.2f\n", a / b}')
        echo "Effective_Tokens_per_second_per_gpu: ${Effective_Tokens_per_second_per_gpu}" >> ${log_file}
        Train_samples_per_second=`cat ${log_file} | grep 'train_samples_per_second' \
                                            |awk -F'train_samples_per_second: ' '{print $2}' |awk -F', ' '{print $1}'`
        length=4096
        Total_Tokens_per_second=$(awk -v a="$Train_samples_per_second" -v b="$length" 'BEGIN {printf "%.2f\n", a * b}')
        Total_Tokens_per_second_per_gpu=$(awk -v a="$Total_Tokens_per_second" -v b="$num_gpu" 'BEGIN {printf "%.2f\n", a / b}')
        echo "Total_Tokens_per_second_per_gpu: ${Total_Tokens_per_second_per_gpu}" >> ${log_file}
    fi
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
    else
        echo -e "${model_name}, SUCCESS"
    fi
    #kill -9 `ps -ef|grep 'python'|awk '{print $2}'`
    if [ ${device_num} != "N1C1" -a -d mylog ]; then
        case_path=$PWD && cd - && mkdir -p mylog      # PaddleNLP/tests/mylog
        cp -r ${case_path}/mylog/workerlog.* ./mylog/
    fi
}

export PYTHONPATH=$(dirname "$PWD"):$PYTHONPATH

source ${BENCHMARK_ROOT}/scripts/run_model.sh   # 在该脚本中会对符合benchmark规范的log使用analysis.py 脚本进行性能数据解析;如果不联调只想要产出训练log可以注掉本行,提交时需打开
_set_params $@
#_train       # 如果只产出训练log,不解析,可取消注释
_run     # 该函数在run_model.sh中,执行时会调用_train; 如果不联调只产出训练log可以注掉本行,提交时需打开
