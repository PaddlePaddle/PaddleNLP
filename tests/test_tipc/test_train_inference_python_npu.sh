#!/bin/bash
source test_tipc/common_func.sh

function readlinkf() {
    perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

function func_parser_config() {
    strs=$1
    IFS=" "
    array=(${strs})
    tmp=${array[2]}
    echo ${tmp}
}

function func_parser_dir() {
    strs=$1
    IFS="/"
    array=(${strs})
    len=${#array[*]}
    dir=""
    count=1
    for arr in ${array[*]}; do 
        if [ ${len} = "${count}" ]; then
            continue;
        else
            dir="${dir}/${arr}"
            count=$((${count} + 1))
        fi
    done
    echo "${dir}"
}

BASEDIR=$(dirname "$0")
REPO_ROOT_PATH=$(readlinkf ${BASEDIR}/../)

FILENAME=$1

# change gpu to npu in tipc txt configs
sed -i "s/--device:gpu/--device:npu/g" $FILENAME
sed -i "s/gpu_list:0|0,1/gpu_list:1/g" $FILENAME
sed -i "s/state=GPU/state=NPU|cpu/g" $FILENAME
sed -i "s/trainer:pact_train/trainer:norm_train/g" $FILENAME
sed -i "s/trainer:fpgm_train/trainer:norm_train/g" $FILENAME
sed -i "s/--device:cpu|gpu/--device:cpu|npu/g" $FILENAME
sed -i "s/--device:gpu|cpu/--device:cpu|npu/g" $FILENAME
sed -i "s/--benchmark:True/--benchmark:False/g" $FILENAME
sed -i "s/--use_tensorrt:False|True/--use_tensorrt:False/g" $FILENAME
sed -i 's/\"gpu\"/\"npu\"/g' test_tipc/test_train_inference_python.sh

# parser params
dataline=`cat $FILENAME`
IFS=$'\n'
lines=(${dataline})

# pass parameters to test_train_inference_python.sh
cmd="bash test_tipc/test_train_inference_python.sh ${FILENAME} $2"
echo $cmd
eval $cmd


