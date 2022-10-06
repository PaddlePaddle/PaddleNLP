#!/usr/bin/env bash
# Test training benchmark for a model.
# Usage：bash benchmark/run_benchmark.sh ${model_item} ${fp_item} ${mp_degree} ${pp_degree} ${dp_degree} ${micro_batch_size} ${global_batch_size} ${run_mode} ${device_num} ${use_sharding}
function _set_params(){
    model_item=${1:-"model_item"}   # (必选) 模型 item
    fp_item=${2:-"fp32"}            # (必选) fp32|fp16
    dp_degree=${3:-"1"}             # (必选) dp/moe数据并行度
    sharding_degree=${4:-"1"}       # (必选) sharding/moe分组参数切分并行度
    local_batch_size=${5:-"2"}      # (必选) 每张卡的batch_size
    run_mode=${6:-"DP"}             # (必选) MP模型并行|DP数据并行|PP流水线并行|混合并行DP1-MP1-PP1|DP1-MP4-PP1|DP_MoE_C1|Sharding_MoE_C1
    device_num=${7:-"N1C1"}         # (必选) 使用的卡数量，N1C1|N1C8|N4C32 （4机32卡）
    profiling=${PROFILING:-"false"}      # (必选) Profiling  开关，默认关闭，通过全局变量传递
    model_repo="PaddleNLP"          # (必选) 模型套件的名字
    speed_unit="tokens/s"         # (必选)速度指标单位
    skip_steps=0                  # (必选)解析日志，跳过模型前几个性能不稳定的step
    keyword="ips:"                 # (必选)解析日志，筛选出性能数据所在行的关键字
    convergence_key="loss:"        # (可选)解析日志，筛选出收敛数据所在行的关键字 如：convergence_key="loss:"
    num_experts=${8:-8}             #(可选)每张卡的expert数量
    max_iter=${9:-500}                      # （可选）需保证模型执行时间在5分钟内，需要修改代码提前中断的直接提PR 合入套件；或使用max_epoch参数
    use_sharding=${10:-"false"}               # （可选) 是否使用ShardingOptimizer
    num_workers=0                  # (可选)
    base_batch_size=$local_batch_size
    # 以下为通用执行命令，无特殊可不用修改
    model_name=${model_item}_bs${local_batch_size}_${fp_item}_${run_mode}  # (必填) 且格式不要改动,与竞品名称对齐
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

    # if [ ${model_item} = "gpt3_moe" ];then
    #     static_scripts="../examples/language_model/gpt-moe/dygraph/"
    # else
    #     echo "not supported model item: ${model_item}"; exit 1;
    # fi

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

    # data_path="./data/"

    use_pure_fp16=False
    if [ "fp16" = ${fp_item} ]; then use_pure_fp16=True; fi
    train_cmd="${add_options} \
               --model_type gpt \
                --model_name_or_path gpt2-small-en \
                --input_dir ./data\
                --output_dir output\
                --weight_decay 0.01\
                --grad_clip 2\
                --max_steps ${max_iter}\
                --save_steps 100000\
                --decay_steps 320000\
                --device gpu\
                --eval_freq 100000\
                --warmup_rate 0.01\
                --local_batch_size ${local_batch_size} \
                --micro_batch_size ${local_batch_size} \
                --dp_degree ${dp_degree}\
                --mp_degree 1\
                --pp_degree 1\
                --sharding_degree ${sharding_degree}\
                --expert_mode True\
                --logging_freq 1 \
                --num_experts ${num_experts}\
                --use_pure_fp16 ${use_pure_fp16} \
                --use_recompute False\
                --recompute_partition False\
                --recompute_offload False\
                --scale_loss 32768 \
                --gate gshard \
                --balance_loss_weight 1.0"

    # 以下为通用执行命令，无特殊可不用修改
    case ${run_mode} in
    DP_MoE_C1) echo "run run_mode: DP_MoE_C1"
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --gpus=0 \
              run_moe_pretrain.py ${train_cmd}"
        workerlog_id=0
        ;;
    DP_MoE_C8|DP_MoE_C16) echo "run run_mode: ${run_mode}"
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --gpus=0,1,2,3,4,5,6,7 \
              run_moe_pretrain.py ${train_cmd}"
        workerlog_id=0
        ;;
    DP_MoE_C32) echo "run run_mode: DP_MoE_C32"
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --gpus=0,1,2,3,4,5,6,7 \
              run_moe_pretrain.py ${train_cmd}"
        workerlog_id=0
        ;;
    Sharding_MoE_C8|Sharding_MoE_C16) echo "run run_mode: ${run_mode}"
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --gpus=0,1,2,3,4,5,6,7 \
              run_moe_pretrain.py ${train_cmd}"
        workerlog_id=0
        ;;
    Sharding_MoE_C32) echo "run run_mode: Sharding_MoE_C32"
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --gpus=0,1,2,3,4,5,6,7 \
              run_moe_pretrain.py ${train_cmd}"
        workerlog_id=0
        ;;
    *) echo "choose run_mode "; exit 1;
    esac
    cd ../examples/language_model/moe/dygraph/
    echo "train_cmd: ${train_cmd}  log_file: ${log_file}"
    python -c "import paddlenlp"
    if [[ ${model_item} =~ "CE" ]];then # CE精度-不限制执行时间
        ${train_cmd} > ${log_file} 2>&1
    else
        timeout 15m ${train_cmd} > ${log_file} 2>&1
    fi
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
    else
        echo -e "${model_name}, SUCCESS"
    fi
    #kill -9 `ps -ef|grep 'python'|awk '{print $2}'`
    if [ ${device_num} != "N1C1" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.${workerlog_id} ${log_file}
    fi
}

export PYTHONPATH=$(dirname "$PWD"):$PYTHONPATH

source ${BENCHMARK_ROOT}/scripts/run_model.sh   # 在该脚本中会对符合benchmark规范的log使用analysis.py 脚本进行性能数据解析;如果不联调只想要产出训练log可以注掉本行,提交时需打开
_set_params $@
#_train       # 如果只产出训练log,不解析,可取消注释
_run     # 该函数在run_model.sh中,执行时会调用_train; 如果不联调只产出训练log可以注掉本行,提交时需打开
