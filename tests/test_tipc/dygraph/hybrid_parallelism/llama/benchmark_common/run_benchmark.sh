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
    model_name_or_path=${model_name_or_path:-"facebook/llama-13b"}
    per_device_train_batch_size=${per_device_train_batch_size:-1}
    tensor_parallel_degree=${tensor_parallel_degree:-2}
    data_parallel_degree=${data_parallel_degree:-1}
    pipeline_parallel_degree=${pipeline_parallel_degree:-4}
    virtual_pp_degree=${virtual_pp_degree:-2}
    sequence_parallel=${sequence_parallel:-0}
    sharding_parallel_degree=${sharding_parallel_degree:-2}
    sharding=${sharding:-"stage1"}
    recompute=${recompute:-1}
    run_mode=${run_mode:-"DP1-MP2-PP4-mbs1-acc32-recompute"}
    device_num=${device_num:-"N2C16"}
    global_batch_size=${global_batch_size:-64}
    model_item=${model_item:-"facebook-llama-13b_seqlen2048_pretrain"}
    max_steps=${max_steps:-150}
    gradient_accumulation_steps=${gradient_accumulation_steps:-32}
    pp_recompute_interval=${pp_recompute_interval:-1}
    tensor_parallel_config=${tensor_parallel_config:-"enable_mp_async_allreduce,enable_mp_skip_c_identity,enable_mp_fused_linear_param_grad_add"}
    pipeline_parallel_config=${pipeline_parallel_config:-""}
    recompute_use_reentrant=${recompute_use_reentrant:-"true"}

    base_batch_size=${global_batch_size}

    profiling=${PROFILING:-"false"}      # (必选) Profiling  开关，默认关闭，通过全局变量传递
    model_repo="PaddleNLP"          # (必选) 模型套件的名字
    speed_unit="tokens/s"         # (必选)速度指标单位
    skip_steps=0                  # (必选)解析日志，跳过模型前几个性能不稳定的step
    keyword="ips:"                 # (必选)解析日志，筛选出性能数据所在行的关键字

    convergence_key="loss:"        # (可选)解析日志，筛选出收敛数据所在行的关键字 如：convergence_key="loss:"

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

    if [ $fp_item = "fp16" ]; then
        use_fp16_cmd="--use_amp true"
    fi

    # Disable for hanging bug
    # if [ "${tensor_parallel_degree}" != "1" ]; then
    #     export CUDA_DEVICE_MAX_CONNECTIONS=1
    # fi

    if [ "${pipeline_parallel_config}" != "" ]; then
        pipeline_parallel_config_args="--pipeline_parallel_config ${pipeline_parallel_config}"
    else
        pipeline_parallel_config_args=""
    fi

    use_pure_fp16=False
    train_cmd="--model_name_or_path ${model_name_or_path} \
    --tokenizer_name_or_path ${model_name_or_path} \
    --input_dir ./data \
    --output_dir ./output \
    --split 949,50,1 \
    --max_seq_length 2048 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --use_flash_attention 1 \
    --use_fused_rms_norm 1 \
    --bf16  \
    --fp16_opt_level O2  \
    --amp_master_grad \
    --tensor_parallel_degree ${tensor_parallel_degree} \
    --pipeline_parallel_degree ${pipeline_parallel_degree} \
    --virtual_pp_degree ${virtual_pp_degree} \
    --pp_recompute_interval ${pp_recompute_interval} \
    --learning_rate 0.00001 \
    --min_learning_rate 0.000001 \
    --max_steps ${max_steps} \
    --save_steps 50000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --dataloader_num_workers 1 \
    --eval_steps 1001 \
    --sharding ${sharding} \
    --disable_tqdm true \
    --continue_training 0 \
    --do_train \
    --device gpu \
    --use_fused_rope true \
    --enable_linear_fused_grad_add true \
    --fuse_attention_qkv true \
    --fuse_attention_ffn true \
    --tensor_parallel_config ${tensor_parallel_config} ${pipeline_parallel_config_args} \
    --recompute ${recompute} \
    --recompute_use_reentrant ${recompute_use_reentrant} \
    --skip_memory_metrics 0 \
    --data_cache ./data_cache"

    if [ ${PADDLE_TRAINER_ID} ]
    then
        PADDLE_RANK_OPTION=" --rank ${PADDLE_TRAINER_ID}"
    else
        PADDLE_RANK_OPTION=""
    fi
    # 以下为通用执行命令，无特殊可不用修改
    case ${device_num} in
    N1C1) echo "Run with: device_num=${device_num}, run_mode=${run_mode}"
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --gpus=0 ${PADDLE_RANK_OPTION}\
            run_pretrain.py ${train_cmd}"
        workerlog_id=0
        ;;
    N1C2) echo "Run with: device_num=${device_num}, run_mode=${run_mode}"
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --gpus=0,1 ${PADDLE_RANK_OPTION}\
              run_pretrain.py ${train_cmd}" 
        workerlog_id=0
        ;;
    N1C4) echo "Run with: device_num=${device_num}, run_mode=${run_mode}"
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --gpus=0,1,2,3 ${PADDLE_RANK_OPTION}\
            run_pretrain.py ${train_cmd}"
        workerlog_id=0
        ;;
    *) echo "Run with: device_num=${device_num}, run_mode=${run_mode}"
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --gpus=0,1,2,3,4,5,6,7 ${PADDLE_RANK_OPTION}\
            run_pretrain.py ${train_cmd}"
        workerlog_id=0
        ;;
    esac
    cd ../llm
    echo "train_cmd: ${train_cmd}  log_file: ${log_file}"
    python -c "import paddlenlp"
    if [[ ${model_name_or_path} =~ "CE" ]];then # CE精度-不限制执行时间
        ${train_cmd} > ${log_file} 2>&1
    else
        timeout 30m ${train_cmd} > ${log_file} 2>&1
        # echo ${train_cmd}
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
        rm ${log_file}
        cp ${case_path}/mylog/workerlog.${workerlog_id} ${log_file}
    fi
}

export PYTHONPATH=$(dirname "$PWD"):$PYTHONPATH

source ${BENCHMARK_ROOT}/scripts/run_model.sh   # 在该脚本中会对符合benchmark规范的log使用analysis.py 脚本进行性能数据解析;如果不联调只想要产出训练log可以注掉本行,提交时需打开
_set_params $@
#_train       # 如果只产出训练log,不解析,可取消注释
_run     # 该函数在run_model.sh中,执行时会调用_train; 如果不联调只产出训练log可以注掉本行,提交时需打开
