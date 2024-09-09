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
# Usage：bash benchmark/run_benchmark.sh ${model_item} ${fp_item} ${mp_degree} ${pp_degree} ${dp_degree} ${micro_batch_size} ${global_batch_size} ${run_mode} ${device_num} ${use_sharding}
function _set_params(){
    model_item=${1:-"llama-7b_auto_dp2mp2pp2"}   # (必选) 模型 item
    base_batch_size=${2:-"1"}       # (必选)
    fp_item=${3:-"fp32"}            # (必选) fp32|fp16|bf16
    run_mode=${4:-"DP"}             # (必选) MP模型并行|DP数据并行|PP流水线并行|混合并行DP1-MP1-PP1|DP1-MP4-PP1
    device_num=${5:-"N1C1"}         # (必选) 使用的卡数量，N1C1|N1C8|N4C32 （4机32卡）
    profiling=${PROFILING:-"false"}      # (必选) Profiling  开关，默认关闭，通过全局变量传递

    model_repo="PaddleNLP"          # (必选) 模型套件的名字
    speed_unit="sample/s"         # (必选)速度指标单位
    skip_steps=0                  # (必选)解析日志，跳过模型前几个性能不稳定的step
    keyword="ips:"                 # (必选)解析日志，筛选出性能数据所在行的关键字
    convergence_key="loss:"        # (可选)解析日志，筛选出收敛数据所在行的关键字 如：convergence_key="loss:"
    max_iter=${6:-100}                      # （可选）需保证模型执行时间在5分钟内，需要修改代码提前中断的直接提PR 合入套件；或使用max_epoch参数
    num_workers=0                  # (可选)

    # 以下为通用执行命令，无特殊可不用修改
    model_name=${model_item}_bs${base_batch_size}_${fp_item}_${run_mode}  # (必填) 且格式不要改动,与竞品名称对齐
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}  # （必填） TRAIN_LOG_DIR  benchmark框架设置该参数为全局变量
    profiling_log_path=${PROFILING_LOG_DIR:-$(pwd)}  # （必填） PROFILING_LOG_DIR benchmark框架设置该参数为全局变量
    speed_log_path=${LOG_PATH_INDEX_DIR:-$(pwd)}

    train_log_file=${run_log_path}/${model_repo}_${model_name}_${device_num}_d2sT_log
    profiling_log_file=${profiling_log_path}/${model_repo}_${model_name}_${device_num}_d2sT_profiling
    speed_log_file=${speed_log_path}/${model_repo}_${model_name}_${device_num}_d2sT_speed
}
function _train(){
    batch_size=${base_batch_size}  # 如果模型跑多卡单进程时,请在_train函数中计算出多卡需要的bs
    echo "current CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, model_name=${model_name}, device_num=${device_num}, is profiling=${profiling}"

    if [ ${profiling} = "true" ];then
        add_options="--profiler_options=\"batch_range=[10,20];state=GPU;tracer_option=Default;profile_path=model.profile\""
        log_file=${profiling_log_file}
    else
        add_options=""
        log_file=${train_log_file}
    fi

    export FLAGS_group_schedule_tiling_first=1
    export FLAGS_cinn_bucket_compile=1
    export FLAGS_cinn_new_cluster_op_method=1
    export FLAGS_deny_cinn_ops="gather"
    export FLAGS_prim_forward_blacklist="pd_op.embedding;pd_op.squared_l2_norm"
    export FLAGS_enable_prim_after_distribute=True
    export FLAGS_disable_dyshape_in_train=True
    export FLAGS_enable_pir_in_executor=True
    export FLAGS_enable_prim_after_distribute=1


    export ENABLE_FALL_BACK=True # 开启SOT
    # export FLAGS_use_cinn=True  # 是否开启cinn ,在benchmark中设置

    use_fp16_cmd=""
    if [ $fp_item = "fp16" ]; then
        use_fp16_cmd="--fp16 1 --fp16_opt_level O2"
    fi
    to_static=1  # 是否开启动转静训练
    train_cmd="run_pretrain_auto.py \
            --model_type "llama" \
            --model_name_or_path "facebook/llama-7b" \
            --tokenizer_name_or_path "facebook/llama-7b" \
            --input_dir "./data" \
            --output_dir "output/$model_item" \
            --split 949,50,1 \
            --max_seq_length 2048 \
            --per_device_train_batch_size ${batch_size} \
            --per_device_eval_batch_size 2 \
            --gradient_accumulation_steps 1 \
            --use_flash_attention 1 \
            --use_fused_rms_norm 0 \
            --scale_loss 1024 \
            --pipeline_parallel_degree 1 \
            --tensor_parallel_degree 1 \
            --sharding_parallel_degree 1 \
            --learning_rate 0.0001 \
            --min_learning_rate 0.00001 \
            --max_steps ${max_iter} \
            --save_steps 5000000 \
            --weight_decay 0.01 \
            --warmup_ratio 0.01 \
            --logging_steps 20 \
            --dataloader_num_workers 1 \
            --sharding '' \
            --eval_steps 1000000 \
            --disable_tqdm true \
            --continue_training 0\
            --recompute 0 \
            --do_train \
            --do_eval \
            --device gpu \
            --data_impl mmap \
            --enable_auto_parallel 1 \
            --max_grad_norm 1.0 \
            --num_hidden_layers 4 \
            --to_static ${to_static} \
            ${use_fp16_cmd} "

    # 以下为通用执行命令，无特殊可不用修改
    case ${run_mode} in
    DP)
        rm -rf ./mylog   # 注意执行前删掉log目录
        rm -rf output/$model_item
        train_cmd="python -u -m paddle.distributed.launch --log_dir=./mylog \
            --gpus $CUDA_VISIBLE_DEVICES ${train_cmd}"
        ;;
    DP1-MP1-PP1)  echo "run run_mode: DP1-MP1-PP1" ;;
    *) echo "choose run_mode "; exit 1;
    esac

    cd ../llm/auto_parallel/llama/
    rm -rf ./mylog   # 注意执行前删掉log目录
    rm -rf output/$model_item
    echo "train_cmd: ${train_cmd}  log_file: ${log_file}"

    python -c "import paddlenlp"
    timeout 15m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
    else
        echo -e "${model_name}, SUCCESS"
    fi
    #kill -9 `ps -ef|grep 'python'|awk '{print $2}'`
    if [ -d mylog ]; then
        case_path=$PWD && cd - && mkdir -p mylog      # PaddleNLP/tests/mylog
        cp -r ${case_path}/mylog/workerlog.* ./mylog/
        rm ${log_file}
        cp ${case_path}/mylog/workerlog.0 ${log_file}
    fi

    echo ${train_cmd} >> ${log_file}
    cat ${log_file}
}

function _analysis_log(){
    # PaddleNLP/tests 目录
    analysis_log_cmd="python test_tipc/dygraph/llama-7b_auto_dp2mp2pp2/benchmark_common/analysis_log.py \
        ${model_item} ${log_file} ${speed_log_file} ${device_num} ${base_batch_size} ${fp_item}"
    echo ${analysis_log_cmd}
    eval ${analysis_log_cmd}
}

_set_params $@
str_tmp=$(echo `pip list|grep paddlepaddle-gpu|awk -F ' ' '{print $2}'`)
export frame_version=${str_tmp%%.post*}
export frame_commit=$(echo `python -c "import paddle;print(paddle.version.commit)"`)
export model_branch=`git symbolic-ref HEAD 2>/dev/null | cut -d"/" -f 3`
export model_commit=$(git log|head -n1|awk '{print $2}')
echo "---------frame_version is ${frame_version}"
echo "---------Paddle commit is ${frame_commit}"
echo "---------Model commit is ${model_commit}"
echo "---------model_branch is ${model_branch}"

job_bt=`date '+%Y%m%d%H%M%S'`
_train
job_et=`date '+%Y%m%d%H%M%S'`
export model_run_time=$((${job_et}-${job_bt}))
_analysis_log
