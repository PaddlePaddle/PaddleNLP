#!/usr/bin/env bash
# Test training benchmark for a model.
# Usage：bash benchmark/run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num}
function _set_params(){
    model_item=${1:-"model_item"}   # (必选) 模型 item
    base_batch_size=${2:-"2"}       # (必选) 如果是静态图单进程，则表示每张卡上的BS，需在训练时*卡数
    fp_item=${3:-"fp32"}            # (必选) fp32|fp16
    run_mode=${4:-"DP"}             # (必选) MP模型并行|DP数据并行|PP流水线并行|混合并行DP1-MP1-PP1|DP1-MP4-PP1
    device_num=${5:-"N1C1"}         # (必选) 使用的卡数量，N1C1|N1C8|N4C32 （4机32卡）
    profiling=${PROFILING:-"false"}      # (必选) Profiling  开关，默认关闭，通过全局变量传递
    model_repo="PaddleNLP"          # (必选) 模型套件的名字
    speed_unit="sequences/sec"         # (必选)速度指标单位
    skip_steps=10                  # (必选)解析日志，跳过模型前几个性能不稳定的step
    keyword="ips:"                 # (必选)解析日志，筛选出性能数据所在行的关键字
    convergence_key="loss:"        # (可选)解析日志，筛选出收敛数据所在行的关键字 如：convergence_key="loss:"
    max_iter=500                   # （可选）需保证模型执行时间在5分钟内，需要修改代码提前中断的直接提PR 合入套件；或使用max_epoch参数
    num_workers=0                  # (可选)
    # 以下为通用执行命令，无特殊可不用修改
    model_name=${model_item}_bs${base_batch_size}_${fp_item}_${run_mode}  # (必填) 且格式不要改动,与竞品名称对齐
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
}

function _train(){
    batch_size=${base_batch_size}  # 如果模型跑多卡单进程时,请在_train函数中计算出多卡需要的bs
    static_scripts="../examples/language_model/bert/static/"

    gradient_merge_steps=$(expr 67584 \/ $batch_size \/ 8)

    echo "current CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, model_name=${model_name}, device_num=${device_num}, is profiling=${profiling}"

    if [ ${profiling} = "true" ];then
        add_options="--profiler_options=\"batch_range=[10,20];state=GPU;tracer_option=Default;profile_path=model.profile\""
        log_file=${profiling_log_file}
    else
        add_options=""
        log_file=${train_log_file}
    fi

    OLD_IFS=$IFS
    IFS="_"
    array=(${model_item})

    model_type=${array[1]}
    seq_len=${array[2]}

    IFS=$OLD_IFS

    if [ ${fp_item} == "fp16" ]; then
        fuse_transformer="True"
        use_amp="True"
    elif [ ${fp_item} == "fp32" ]; then
        fuse_transformer="False"
        use_amp="False"
    else
        echo " The fp_item should be fp32 or fp16"
    fi

    if [[ ${model_type} = "base" ]]; then
        train_cmd="${add_options} \
            --max_predictions_per_seq 80 \
            --learning_rate 5e-5 \
            --weight_decay 0.0 \
            --adam_epsilon 1e-8 \
            --warmup_steps 0 \
            --output_dir ./tmp2/ \
            --logging_steps 10 \
            --save_steps 20000 \
            --max_steps ${max_iter} \
            --input_dir ./data/wikicorpus_en_${seq_len} \
            --model_type bert \
            --model_name_or_path bert-${model_type}-uncased \
            --batch_size ${batch_size} \
            --use_amp ${use_amp} \
            --gradient_merge_steps $gradient_merge_steps \
            --fuse_transformer true "
    elif [[ ${model_type} = "large" ]]; then
         train_cmd="${add_options} \
            --max_predictions_per_seq 80 \
            --learning_rate 1e-5 \
            --weight_decay 1e-2 \
            --adam_epsilon 1e-6 \
            --warmup_steps 10000 \
            --output_dir ./tmp2/ \
            --logging_steps 10 \
            --save_steps 20000 \
            --max_steps ${max_iter} \
            --input_dir ./data/wikicorpus_en_${seq_len} \
            --model_type bert \
            --model_name_or_path bert-${model_type}-uncased \
            --batch_size ${batch_size} \
            --use_amp ${use_amp} \
            --gradient_merge_steps $gradient_merge_steps \
            --fuse_transformer ${fuse_transformer} "
    else
        echo "Model type must be base or large. "
    fi

    # 以下为通用执行命令，无特殊可不用修改
    case ${run_mode} in
    DP) if [[ ${device_num} = "N1C1" ]];then
            echo "run ${run_mode} "
            train_cmd="python -u ${static_scripts}/run_pretrain.py ${train_cmd}" 
        else
            rm -rf ./mylog
            train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --gpus=$CUDA_VISIBLE_DEVICES \
                  ${static_scripts}/run_pretrain.py ${train_cmd}" 
        fi
        ;;
    DP1-MP1-PP1)  echo "run run_mode: DP1-MP1-PP1" ;;
    *) echo "choose run_mode "; exit 1;
    esac
    echo "train_cmd: ${train_cmd}  log_file: ${log_file}"
    timeout 15m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
    else
        echo -e "${model_name}, SUCCESS"
    fi
    #kill -9 `ps -ef|grep 'python'|awk '{print $2}'`
    if [ ${device_num} != "N1C1" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}
source ${BENCHMARK_ROOT}/scripts/run_model.sh   # 在该脚本中会对符合benchmark规范的log使用analysis.py 脚本进行性能数据解析;如果不联调只想要产出训练log可以注掉本行,提交时需打开
_set_params $@
# _train       # 如果只产出训练log,不解析,可取消注释
_run     # 该函数在run_model.sh中,执行时会调用_train; 如果不联调只产出训练log可以注掉本行,提交时需打开
