#!/bin/bash

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
sed -i "s/--device:gpu/--device:npu/g" $FILENAME
sed -i "s/state=GPU/state=NPU/g" $FILENAME
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


