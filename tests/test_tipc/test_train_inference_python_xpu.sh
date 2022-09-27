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

BASEDIR=$(dirname "$0")
REPO_ROOT_PATH=$(readlinkf ${BASEDIR}/../)

FILENAME=$1

# change gpu to npu in tipc txt configs
sed -i "s/--device:gpu/--device:xpu/g" $FILENAME
sed -i "s/state=GPU/state=XPU/g" $FILENAME
sed -i "s/trainer:pact_train/trainer:norm_train/g" $FILENAME
sed -i "s/trainer:fpgm_train/trainer:norm_train/g" $FILENAME
sed -i "s/--device:cpu|gpu/--device:cpu|xpu/g" $FILENAME
sed -i "s/--device:gpu|cpu/--device:cpu|xpu/g" $FILENAME
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


