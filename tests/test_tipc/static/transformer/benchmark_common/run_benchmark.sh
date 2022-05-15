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
    speed_unit="words/sec"         # (必选)速度指标单位
    skip_steps=5                  # (必选)解析日志，跳过模型前几个性能不稳定的step
    keyword="ips:"                 # (必选)解析日志，筛选出性能数据所在行的关键字
    convergence_key="loss:"        # (可选)解析日志，筛选出收敛数据所在行的关键字 如：convergence_key="loss:"
    max_iter=${6:-500}                     # （可选）需保证模型执行时间在5分钟内，需要修改代码提前中断的直接提PR 合入套件；或使用max_epoch参数
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
    static_scripts="../examples/machine_translation/transformer/static/"

    echo "current CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, model_name=${model_name}, device_num=${device_num}, is profiling=${profiling}"

    if [ ${model_item} == "transformer_base" ]; then
        config_file="transformer.base.yaml"
    elif [ ${model_item} == "transformer_big" ]; then
        config_file="transformer.big.yaml"
    else
        echo "The model should be transformer_big or transformer_base!"
        exit 1
    fi

    if [ ${profiling} = "true" ];then
        add_options="--profiler_options=\"batch_range=[10,20];state=GPU;tracer_option=Default;profile_path=model.profile\""
        log_file=${profiling_log_file}
    else
        add_options=""
        log_file=${train_log_file}
    fi

    if [ ${fp_item} == "amp_fp16" ]; then
        sed -i "s/^use_amp.*/use_amp: True/g" ${static_scripts}/../configs/${config_file}
        sed -i "s/^use_pure_fp16.*/use_pure_fp16: False/g" ${static_scripts}/../configs/${config_file}
    elif [ ${fp_item} == "pure_fp16" ]; then
        sed -i "s/^use_amp.*/use_amp: True/g" ${static_scripts}/../configs/${config_file}
        sed -i "s/^use_pure_fp16.*/use_pure_fp16: True/g" ${static_scripts}/../configs/${config_file}
    elif [ ${fp_item} == "fp32" ]; then
        sed -i "s/^use_amp.*/use_amp: False/g" ${static_scripts}/../configs/${config_file}
        sed -i "s/^use_pure_fp16.*/use_pure_fp16: False/g" ${static_scripts}/../configs/${config_file}
    else
        echo " The fp_item should be fp32 pure_fp16 or amp_fp16"
    fi

    sed -i "s/^max_iter.*/max_iter: ${max_iter}/g" ${static_scripts}/../configs/${config_file}
    sed -i "s/^batch_size:.*/batch_size: ${base_batch_size}/g" ${static_scripts}/../configs/${config_file}

    train_cmd="${add_options} --config ${static_scripts}/../configs/${config_file} --train_file ${static_scripts}/../train.en ${static_scripts}/../train.de --dev_file ${static_scripts}/../dev.en ${static_scripts}/../dev.de --vocab_file ${static_scripts}/../vocab_all.bpe.33712 --unk_token <unk> --bos_token <s> --eos_token <e> --benchmark"

    # 以下为通用执行命令，无特殊可不用修改
    case ${run_mode} in
    DP) if [[ ${device_num} = "N1C1" ]];then
            echo "run ${run_mode} "
            sed -i "s/^is_distributed.*/is_distributed: False/g" ${static_scripts}/../configs/${config_file}
            train_cmd="python -u ${static_scripts}/train.py ${train_cmd}" 
        else
            sed -i "s/^is_distributed.*/is_distributed: True/g" ${static_scripts}/../configs/${config_file}
            rm -rf ./mylog
            train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --gpus=$CUDA_VISIBLE_DEVICES \
                  ${static_scripts}/train.py --distributed ${train_cmd}" 
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
