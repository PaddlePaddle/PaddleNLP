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
# Usage：bash benchmark/run_benchmark.sh ${model_name_or_path} ${per_device_train_batch_size} ${use_flash_attention} ${tensor_parallel_degree} ${pipeline_parallel_degree} ${virtual_pp_degree} ${sequence_parallel} ${sharding_degree} ${num_train_epochs} ${save_steps} ${sharding} ${recompute} ${run_mode} ${device_num}
function _set_params(){
    # 脚本所需参数
    model_name_or_path=${1:-"facebook/llama-7b"}
    dataset_name_or_path=${2:-"llm_benchmark_zh"}
    base_batch_size=${3:-"1"}
    learning_rate=${4:-"3e-05"}
    recompute=${5:-"true"}
    tensor_parallel_degree=${6:-"1"}
    lora=${7:-"false"}
    prefix_tuning=${8:-"false"}

    # benchmark配置参数
    model_item=${9:-"facebook/llama-7b"}   # (必选) 模型 item |fastscnn|segformer_b0| ocrnet_hrnetw48
    fp_item="fp16"            # (必选) fp32|fp16
    run_mode=${10:-"DP"}             # (必选) MP模型并行|DP数据并行|PP流水线并行|混合并行DP1-MP1-PP1|DP1-MP4-PP1
    device_num=${11:-"N1C1"}         # (必选) 使用的卡数量，N1C1|N1C8|N4C32 （4机32卡）
    profiling="false"      # (必选) Profiling  开关，默认关闭，通过全局变量传递
    num_train_epochs=${12:-"1"}

    model_repo="PaddleNLP"          # (必选) 模型套件的名字
    speed_unit="tokens/s"         # (必选)速度指标单位
    skip_steps=0                  # (必选)解析日志，跳过模型前几个性能不稳定的step
    keyword="Effective_Tokens_per_second:"                 # (必选)解析日志，筛选出性能数据所在行的关键字
    convergence_key="train_loss:"        # (可选)解析日志，筛选出收敛数据所在行的关键字 如：convergence_key="loss:"
    is_large_model=True           # (可选)普通模型默认为False，如果添加大模型且只取一条ips设置为True

    # 以下为通用执行命令，无特殊可不用修改
    model_name=${model_item}_bs${base_batch_size}_${fp_item}_${run_mode}  # (必填) 且格式不要改动,与竞品名称对齐
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
    batch_size=1 # 如果模型跑多卡单进程时,请在_train函数中计算出多卡需要的bs

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

    if [ $fp_item = "fp16" ]; then
        use_fp16_cmd="--use_amp true"
    fi

    use_pure_fp16=False

    train_cmd="    --model_name_or_path ${model_name_or_path} \
            --dataset_name_or_path ${dataset_name_or_path} \
            --output_dir output \
            --per_device_train_batch_size ${base_batch_size} \
            --gradient_accumulation_steps 1 \
            --num_train_epochs ${num_train_epochs} \
            --learning_rate ${learning_rate} \
            --warmup_steps 30 \
            --evaluation_strategy no \
            --save_strategy no \
            --logging_steps 1 \
            --src_length 1024 \
            --max_length 1024 \
            --fp16 1 \
            --fp16_opt_level O2 \
            --do_train 1 \
            --do_eval 0 \
            --disable_tqdm 1 \
            --eval_with_do_generation 0 \
            --recompute ${recompute} \
            --tensor_parallel_degree ${tensor_parallel_degree} \
            --lora ${lora} \
            --prefix_tuning ${prefix_tuning} \
            --benchmark 1 \
            --zero_padding 1 \
            --device gpu"

    # 以下为通用执行命令，无特殊可不用修改
    cd ../llm/
    echo "run run_mode: ${run_mode} device_num: ${device_num}"
    if [ "N1C1" = ${device_num} ]; then
        train_cmd="python -u run_finetune.py ${train_cmd}" 
    else
        rm -rf ./mylog   # 注意执行前删掉log目录
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --gpus=$CUDA_VISIBLE_DEVICES \
            run_finetune.py ${train_cmd}" 
    fi

    echo "train_cmd: ${train_cmd}  log_file: ${log_file}"

    python -c "import paddlenlp"
    ${train_cmd} > ${log_file} 2>&1

    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
    else
        echo -e "${model_name}, SUCCESS"
    fi
    #kill -9 `ps -ef|grep 'python'|awk '{print $2}'`
    if [ ${device_num} != "N1C1" -a -d mylog ]; then
        case_path=$PWD && cd - && mkdir -p mylog      # PaddleNLP/tests/mylog
        cp -r ${case_path}/mylog/workerlog.* ./mylog/
        rm ${log_file}
        cp ${case_path}/mylog/workerlog.0 ${log_file}
    fi
}

export PYTHONPATH=$(dirname "$PWD"):$PYTHONPATH

source ${BENCHMARK_ROOT}/scripts/run_model.sh   # 在该脚本中会对符合benchmark规范的log使用analysis.py 脚本进行性能数据解析;如果不联调只想要产出训练log可以注掉本行,提交时需打开
_set_params $@
# _train       # 如果只产出训练log,不解析,可取消注释
_run     # 该函数在run_model.sh中,执行时会调用_train; 如果不联调只产出训练log可以注掉本行,提交时需打开