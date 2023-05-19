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

IFS=$'\n'
BASE_CONFIG_FILE=$1
# always use the lite_train_lite_infer mode to speed. Modify the config file.
MODE=lite_train_lite_infer
BASEDIR=$(dirname "$0")

# get the log path.
dataline=$(cat ${BASE_CONFIG_FILE})
lines=(${dataline})
model_name=$(func_parser_value "${lines[1]}")
LOG_PATH="./test_tipc/output/${model_name}/${MODE}"
rm -rf $LOG_PATH
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results_python.log"

# make cudnn algorithm deterministic, such as conv.
export FLAGS_cudnn_deterministic=True

# read the base config and parse and run the sub commands
config_line_numbers=`cat ${BASE_CONFIG_FILE} | grep -n "============" | cut -d':' -f1`
for cln in $config_line_numbers
do
    # change IFS to prevent \n is parsed as delimiter.
    IFS=""
    config_lines=$(cat ${BASE_CONFIG_FILE} | sed -n "${cln},\$p" | head -n 22)
    config_name=`echo ${config_lines} | grep '=====' | cut -d' ' -f2`
    FILENAME=$LOG_PATH/dy2static_$config_name.txt
    echo "[Start dy2static]" "${config_name} : ${FILENAME}"
    echo ${config_lines} > $FILENAME
    sed -i 's/gpu_list.*$/gpu_list:0/g' $FILENAME

    # execute the last line command
    custom_cmd=$(echo $config_lines | tail -n 1)
    echo "CustomCmd is: " $custom_cmd
    eval $custom_cmd

    IFS=$'\n'

    # start dygraph train
    dygraph_output=$LOG_PATH/${config_name}_python_train_infer_dygraph_output.txt
    dygraph_loss=$LOG_PATH/${config_name}_dygraph_loss.txt
    cmd="bash test_tipc/test_train_inference_python.sh ${FILENAME} $MODE >$dygraph_output 2>&1"
    echo $cmd
    eval $cmd

    # start dy2static train
    dy2static_output=$LOG_PATH/${config_name}_python_train_infer_dy2static_output.txt
    dy2static_loss=$LOG_PATH/${config_name}_dy2static_loss.txt
    sed -i '16s/$/ --to_static/g' ${FILENAME}
    cmd="bash test_tipc/test_train_inference_python.sh ${FILENAME} $MODE >$dy2static_output 2>&1"
    echo $cmd
    eval $cmd

    # analysis and compare the losses. 
    dyout=`cat $dy2static_output | python test_tipc/extract_loss.py -v 'step_idx' -e 'avg loss: {%f}'`
    stout=`cat $dygraph_output   | python test_tipc/extract_loss.py -v 'step_idx' -e 'avg loss: {%f}'`
    echo $dyout > $dygraph_loss
    echo $stout > $dy2static_loss
    diff_log=$LOG_PATH/${config_name}_diff_log.txt
    diff_cmd="diff -w $dygraph_loss $dy2static_loss > $diff_log"
    eval $diff_cmd
    last_status=$?
    cat $diff_log
    if [ "$dyout" = "" ]; then
        status_check 1 $diff_cmd $status_log $model_name $diff_log
    elif [ "$stout" = "" ]; then
        status_check 2 $diff_cmd $status_log $model_name $diff_log
    else
        status_check $last_status $diff_cmd $status_log $model_name $diff_log
    fi
done

