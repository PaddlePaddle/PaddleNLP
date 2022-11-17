#!/bin/bash
source test_tipc/common_func.sh

# always use the lite_train_lite_infer mode to speed. Modify the config file.
MODE=lite_train_lite_infer
BASEDIR=$(dirname "$0")

FILENAME=$1
sed -i 's/gpu_list.*$/gpu_list:0/g' $FILENAME
sed -i '23,$d' $FILENAME


# get the log path.
IFS=$'\n'
dataline=$(cat ${FILENAME})
lines=(${dataline})
model_name=$(func_parser_value "${lines[1]}")
LOG_PATH="./test_tipc/output/${model_name}/${MODE}"
rm -rf $LOG_PATH
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results_python.log"

# make cudnn algorithm deterministic, such as conv.
export FLAGS_cudnn_deterministic=True

# start dygraph train
dygraph_output=$LOG_PATH/python_train_infer_dygraph_output.txt
dygraph_loss=$LOG_PATH/dygraph_loss.txt
sed -i '15ctrainer:norm_train' ${FILENAME}
cmd="bash test_tipc/test_train_inference_python.sh ${FILENAME} $MODE >$dygraph_output 2>&1"
echo $cmd
eval $cmd

# start dy2static train
dy2static_output=$LOG_PATH/python_train_infer_dy2static_output.txt
dy2static_loss=$LOG_PATH/dy2static_loss.txt
sed -i '15ctrainer:to_static_train' ${FILENAME}
cmd="bash test_tipc/test_train_inference_python.sh ${FILENAME} $MODE >$dy2static_output 2>&1"
echo $cmd
eval $cmd

# analysis and compare the losses. 
dyout=`cat $dy2static_output | python test_tipc/extract_loss.py -v 'step_idx' -e 'avg loss: {%f}'`
stout=`cat $dygraph_output   | python test_tipc/extract_loss.py -v 'step_idx' -e 'avg loss: {%f}'`
echo $dyout > $dygraph_loss
echo $stout > $dy2static_loss
diff_log=$LOG_PATH/diff_log.txt
diff_cmd="diff -w $dygraph_loss $dy2static_loss | tee $diff_log"
eval $diff_cmd
last_status=$?
if [ "$dyout" = "" ]; then
    status_check 2 $diff_cmd $status_log $model_name $diff_log
fi
if [ "$stout" = "" ]; then
    status_check 2 $diff_cmd $status_log $model_name $diff_log
fi
status_check $last_status $diff_cmd $status_log $model_name $diff_log

