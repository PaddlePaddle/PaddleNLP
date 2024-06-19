#!/usr/bin/env bash

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
# Usage：bash benchmark/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_batch_size} ${global_batch_size} ${run_mode} ${device_num} ${use_sharding}
function _set_params(){
    model_item=${1:-"model_item"}   # (必选) 模型 item
    fp_item=${2:-"fp32"}            # (必选) fp32|fp16
    dp_degree=${3:-"1"}             # (必选) dp数据并行度
    mp_degree=${4:-"1"}             # (必选) mp数据并行度
    pp_degree=${5:-"1"}             # (必选) pp数据并行度
    micro_batch_size=${6:-"2"}      # (必选) micro_batch_size
    global_batch_size=${7:-"16"}    # （必选）global_batch_size
    run_mode=${8:-"DP"}             # (必选) MP模型并行|DP数据并行|PP流水线并行|混合并行DP1-MP1-PP1|DP2-MP8-PP2|DP1-MP8-PP4|DP4-MP8-PP1
    device_num=${9:-"N1C1"}         # (必选) 使用的卡数量，N1C1|N1C8|N4C32 （4机32卡）
    profiling=${PROFILING:-"false"}      # (必选) Profiling  开关，默认关闭，通过全局变量传递
    model_repo="PaddleNLP"          # (必选) 模型套件的名字
    speed_unit="tokens/s"         # (必选)速度指标单位
    skip_steps=0                  # (必选)解析日志，跳过模型前几个性能不稳定的step
    keyword="ips:"                 # (必选)解析日志，筛选出性能数据所在行的关键字
    convergence_key="loss:"        # (可选)解析日志，筛选出收敛数据所在行的关键字 如：convergence_key="loss:"
    sequence_parallel=${10:-"False"}    # (可选)是否打开sequence_parallel
    max_iter=${11:-1000}                      # （可选）需保证模型执行时间在5分钟内，需要修改代码提前中断的直接提PR 合入套件；或使用max_epoch参数
    eval_freq=${12:-"1000"}         # (可选)模型评估间隔
    num_workers=0                  # (可选)
    base_batch_size=$global_batch_size
    use_recompute=${13:-"True"}    # (可选)是否打开recompute
    sharding_degree=${14:-"1"}      # (可选)分组切分并行维度
    sharding_stage=${15:-"1"}       # (可选)切分策略；1表示仅切分优化器状态，2表示再切分梯度，3表示再切分前向参数
    sharding_offload=${16:-"False"} # (可选)CPU offload策略
    # 以下为通用执行命令，无特殊可不用修改
    model_name=${model_item}_bs${global_batch_size}_${fp_item}_${run_mode}  # (必填) 且格式不要改动,与竞品名称对齐
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}  # （必填） TRAIN_LOG_DIR  benchmark框架设置该参数为全局变量
    profiling_log_path=${PROFILING_LOG_DIR:-$(pwd)}  # （必填） PROFILING_LOG_DIR benchmark框架设置该参数为全局变量
    speed_log_path=${LOG_PATH_INDEX_DIR:-$(pwd)}
    #
    train_log_file=${run_log_path}/${model_repo}_${model_name}_${device_num}_log
    profiling_log_file=${profiling_log_path}/${model_repo}_${model_name}_${device_num}_profiling
    speed_log_file=${speed_log_path}/${model_repo}_${model_name}_${device_num}_speed

    OUTPUT_PATH=${run_log_path}/output
}

function _train(){
    batch_size=${local_batch_size}  # 如果模型跑多卡单进程时,请在_train函数中计算出多卡需要的bs

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

    local_batch_size=`expr ${global_batch_size} / ${dp_degree} / ${sharding_degree}`
    num_attention_heads=16 #"gpt2-medium-en"
    if [ ${mp_degree} -lt 8 -a ${pp_degree} -lt 8 ]; then num_attention_heads=4; fi #"gpt2-small-en"
    num_layers=24 #"gpt2-medium-en"
    if [ ${mp_degree} -lt 8 -a ${pp_degree} -lt 8 ]; then num_layers=4; fi #"gpt2-small-en"
    use_pure_fp16=False
    if [ "fp16" = ${fp_item} ]; then use_pure_fp16=True; fi
    train_cmd="-o Engine.max_steps=${max_iter} \
               -o Engine.eval_iters=${eval_freq} \
               -o Distributed.dp_degree=${dp_degree} \
               -o Distributed.mp_degree=${mp_degree} \
               -o Distributed.pp_degree=${pp_degree} \
               -o Distributed.sharding.sharding_degree=${sharding_degree} \
               -o Distributed.sharding.sharding_stage=${sharding_stage} \
               -o Distributed.sharding.sharding_offload=${sharding_offload} \
               -o Model.sequence_parallel=${sequence_parallel}  \
               -o Distributed.sharding.reduce_overlap=False \
               -o Distributed.sharding.broadcast_overlap=False \
               -o Profiler_pretrain.memory_stats=True \
               -o Optimizer.tensor_fusion=False "

    if [ ${PADDLE_TRAINER_ID} ]
    then
        PADDLE_RANK_OPTION=" --rank ${PADDLE_TRAINER_ID}"
    else
        PADDLE_RANK_OPTION=""
    fi
    # 以下为通用执行命令，无特殊可不用修改
    case ${run_mode} in
    DP1-MP1-PP1) echo "run run_mode: DP1-MP1-PP1"
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --devices=0 ${PADDLE_RANK_OPTION}\
            tools/train.py -c ppfleetx/configs/nlp/gpt/pretrain_gpt_1.3B_dp8.yaml \
            ${train_cmd}"
        workerlog_id=0
        ;;
    DP1-MP8-PP1) echo "run run_mode: ${run_mode}"
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 ${PADDLE_RANK_OPTION}\
            tools/train.py -c ppfleetx/configs/nlp/gpt/pretrain_gpt_1.3B_dp8.yaml \
            ${train_cmd}"
        workerlog_id=0
        ;;
    DP2-MP8-PP2) echo "run run_mode: ${run_mode}"
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 ${PADDLE_RANK_OPTION}\
            tools/train.py -c ppfleetx/configs/nlp/gpt/pretrain_gpt_6.7B_sharding16.yaml \
            ${train_cmd}"
        workerlog_id=0
        ;;
    *) echo "choose run_mode "; exit 1;
    esac
    cd ../
    echo "train_cmd: ${train_cmd}  log_file: ${log_file}"
    if [[ ${model_item} =~ "CE" ]];then # CE精度-不限制执行时间
        ${train_cmd} > ${log_file} 2>&1
    else
        timeout 60m ${train_cmd} > ${log_file} 2>&1
    fi
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
    else
        echo -e "${model_name}, SUCCESS"
    fi
    #kill -9 `ps -ef|grep 'python'|awk '{print $2}'`
    if [ ${device_num} != "N1C1" -a -d mylog ]; then
        case_path=$PWD && cd - && mkdir -p mylog      # PaddleNLP/model_zoo/gpt-3/benchmarks
        cp -r ${case_path}/mylog/workerlog.* ./mylog/
        rm ${log_file}
        cp ${case_path}/mylog/workerlog.${workerlog_id} ${log_file}
    fi
}

export PYTHONPATH=$(dirname "$PWD"):$PYTHONPATH

source ${BENCHMARK_ROOT}/scripts/run_model.sh   # 在该脚本中会对符合benchmark规范的log使用analysis.py 脚本进行性能数据解析;如果不联调只想要产出训练log可以注掉本行,提交时需打开
_set_params $@
#_train       # 如果只产出训练log,不解析,可取消注释
_run     # 该函数在run_model.sh中,执行时会调用_train; 如果不联调只产出训练log可以注掉本行,提交时需打开
