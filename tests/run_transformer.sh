#!/bin/bash
FILENAME=$1
# MODE be one of ['lite_train_infer' 'whole_infer' 'whole_train_infer', 'infer']
MODE=$2

dataline=$(cat ${FILENAME})

# parser params
IFS=$'\n'
lines=(${dataline})
function func_parser_key(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[0]}
    echo ${tmp}
}
function func_parser_value(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[1]}
    echo ${tmp}
}
function status_check(){
    last_status=$1   # the exit code
    run_command=$2
    run_log=$3
    if [ $last_status -eq 0 ]; then
        echo -e "\033[33m Run successfully with command - ${run_command}!  \033[0m" | tee -a ${run_log}
    else
        echo -e "\033[33m Run failed with command - ${run_command}!  \033[0m" | tee -a ${run_log}
    fi
}

IFS=$'\n'
# The training params
model_name=$(func_parser_value "${lines[0]}")
python=$(func_parser_value "${lines[1]}")
gpu_list=$(func_parser_value "${lines[2]}")
autocast_list=$(func_parser_value "${lines[3]}")
autocast_key=$(func_parser_key "${lines[3]}")
epoch_key=$(func_parser_key "${lines[4]}")
epoch_num=$(func_parser_value "${lines[4]}")
save_model_key=$(func_parser_key "${lines[5]}")
train_batch_key=$(func_parser_key "${lines[6]}")
train_use_gpu_key=$(func_parser_key "${lines[7]}")
pretrain_model_key=$(func_parser_key "${lines[8]}")
pretrain_model_value=$(func_parser_value "${lines[8]}")

trainer_list=$(func_parser_value "${lines[9]}")
dygraph_trainer=$(func_parser_value "${lines[10]}")
static_trainer=$(func_parser_value "${lines[11]}")
pact_trainer=$(func_parser_value "${lines[12]}")
fpgm_trainer=$(func_parser_value "${lines[13]}")
distill_trainer=$(func_parser_value "${lines[14]}")

dygraph_eval_py=$(func_parser_value "${lines[15]}")
static_eval_py=$(func_parser_value "${lines[16]}")

save_infer_key=$(func_parser_key "${lines[17]}")
export_weight=$(func_parser_key "${lines[18]}")
norm_export=$(func_parser_value "${lines[19]}")
pact_export=$(func_parser_value "${lines[20]}")
fpgm_export=$(func_parser_value "${lines[21]}")
distill_export=$(func_parser_value "${lines[22]}")

inference_py=$(func_parser_value "${lines[23]}")
use_gpu_key=$(func_parser_key "${lines[24]}")
use_gpu_list=$(func_parser_value "${lines[24]}")
use_mkldnn_key=$(func_parser_key "${lines[25]}")
use_mkldnn_list=$(func_parser_value "${lines[25]}")
cpu_threads_key=$(func_parser_key "${lines[26]}")
cpu_threads_list=$(func_parser_value "${lines[26]}")
batch_size_key=$(func_parser_key "${lines[27]}")
batch_size_list=$(func_parser_value "${lines[27]}")
use_trt_key=$(func_parser_key "${lines[28]}")
use_trt_list=$(func_parser_value "${lines[28]}")
precision_key=$(func_parser_key "${lines[29]}")
precision_list=$(func_parser_value "${lines[29]}")
infer_model_key=$(func_parser_key "${lines[30]}")
infer_model=$(func_parser_value "${lines[30]}")
image_dir_key=$(func_parser_key "${lines[31]}")
infer_img_dir=$(func_parser_value "${lines[31]}")
save_log_key=$(func_parser_key "${lines[32]}")

LOG_PATH="./test/output"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results.log"


function func_inference(){
    IFS='|'
    _python=$1
    _script=$2
    _model_dir=$3
    _log_path=$4
    _img_dir=$5

    # inference 
    for use_gpu in ${use_gpu_list[*]}; do 
        if [ ${use_gpu} = "False" ]; then
            for use_mkldnn in ${use_mkldnn_list[*]}; do
                for threads in ${cpu_threads_list[*]}; do
                    for batch_size in ${batch_size_list[*]}; do
                        _save_log_path="${_log_path}/infer_cpu_usemkldnn_${use_mkldnn}_threads_${threads}_batchsize_${batch_size}.log"
                        command="${_python} ${_script} ${use_gpu_key}=${use_gpu} ${use_mkldnn_key}=${use_mkldnn} ${cpu_threads_key}=${threads} ${infer_model_key}=${_model_dir} ${batch_size_key}=${batch_size}"
                        eval $command
                        status_check $? "${command}" "${status_log}"
                    done
                done
            done
        else
            for use_trt in ${use_trt_list[*]}; do
                for precision in ${precision_list[*]}; do
                    if [ ${use_trt} = "False" ] && [ ${precision} != "fp32" ]; then
                        continue
                    fi
                    for batch_size in ${batch_size_list[*]}; do
                        _save_log_path="${_log_path}/infer_gpu_usetrt_${use_trt}_precision_${precision}_batchsize_${batch_size}.log"
                        command="${_python} ${_script} ${use_gpu_key}=${use_gpu} ${infer_model_key}=${_model_dir} ${batch_size_key}=${batch_size}"
                        eval $command
                        status_check $? "${command}" "${status_log}"
                    done
                done
            done
        fi
    done
}

if [ ${MODE} != "infer" ]; then

IFS="|"
for gpu in ${gpu_list[*]}; do
    use_gpu=True
    if [ ${gpu} = "-1" ];then
        use_gpu=False
        env=""
    elif [ ${#gpu} -le 1 ];then
        env="export CUDA_VISIBLE_DEVICES=${gpu}"
        eval ${env}
    elif [ ${#gpu} -le 15 ];then
        IFS=","
        array=(${gpu})
        env="export CUDA_VISIBLE_DEVICES=${array[0]}"
        IFS="|"
    else
        IFS=";"
        array=(${gpu})
        ips=${array[0]}
        gpu=${array[1]}
        IFS="|"
        env=" "
    fi
    for autocast in ${autocast_list[*]}; do 
        for trainer in ${trainer_list[*]}; do 
            if [ ${trainer} = "pact" ]; then
                run_train=${pact_trainer}
                run_export=${pact_export}
            elif [ ${trainer} = "fpgm" ]; then
                run_train=${fpgm_trainer}
                run_export=${fpgm_export}
            elif [ ${trainer} = "distill" ]; then
                run_train=${distill_trainer}
                run_export=${distill_export}
            elif [ ${trainer} = "dygraph" ]; then
                run_train=${dygraph_trainer}
                run_export=${norm_export}
            else
                run_train=${static_trainer}
                run_export=null
            fi

            if [ ${run_train} = "null" ]; then
                continue
            fi

            # not set autocast when autocast is null
            if [ ${autocast} = "null" ]; then
                set_autocast=" "
            else
                set_autocast="${autocast_key}=${autocast}"
            fi
            # not set epoch when whole_train_infer
            if [ ${MODE} != "whole_train_infer" ]; then
                set_epoch="${epoch_key}=${epoch_num}"
            else
                set_epoch=" "
            fi
            # set pretrain
            if [ ${pretrain_model_value} != "null" ]; then
                set_pretrain="${pretrain_model_key}=${pretrain_model_value}"
            else
                set_pretrain=" "
            fi

            save_log="${LOG_PATH}/${trainer}_gpus_${gpu}_autocast_${autocast}"
            if [ ${#gpu} -le 2 ];then  # train with cpu or single gpu
                cmd="${python} ${run_train} ${set_epoch} ${set_pretrain} ${set_autocast}"
            elif [ ${#gpu} -le 15 ];then  # train with multi-gpu
                cmd="${python} -m paddle.distributed.launch --gpus=${gpu} ${run_train} ${set_epoch} ${set_pretrain} ${set_autocast}"
            else     # train with multi-machine
                cmd="${python} -m paddle.distributed.launch --ips=${ips} --gpus=${gpu} ${run_train} ${set_pretrain} ${set_epoch} ${set_autocast}"
            fi
            # run train
            eval $cmd
            status_check $? "${cmd}" "${status_log}"

            if [ ${MODE} = "whole_train_infer" ]
                if [ ${trainer} = "dygraph" ]; then
                    eval_cmd="${python} ${dygraph_eval_py}" 
                elif [ ${trainer} = "static" ]; then
                    eval_cmd="${python} ${static_eval_py}" 
                fi
                # run eval
                eval $eval_cmd
                status_check $? "${eval_cmd}" "${status_log}"
            fi

            if [ ${run_export} = "null" ]; then
                continue
            fi

            # run export model
            save_infer_path="${save_log}"
            export_cmd="${python} ${run_export}"
            eval $export_cmd
            status_check $? "${export_cmd}" "${status_log}"

            #run inference
            eval $env
            save_infer_path="${save_log}"
            func_inference "${python}" "${inference_py}" "${save_infer_path}" "${LOG_PATH}" "${infer_img_dir}"
            eval "unset CUDA_VISIBLE_DEVICES"
        done
    done
done

else
    GPUID=$3
    if [ ${#GPUID} -le 0 ];then
        env=" "
    else
        env="export CUDA_VISIBLE_DEVICES=${GPUID}"
    fi
    echo $env
    #run inference
    func_inference "${python}" "${inference_py}" "${infer_model}" "${LOG_PATH}" "${infer_img_dir}"
fi
