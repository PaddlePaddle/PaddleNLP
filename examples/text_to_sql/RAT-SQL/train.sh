#!/bin/bash

if [ $# -ge 1 ] && [ "$1" == "-h" ]; then
    echo "usage:"
    echo "    $0 trainer_num output_path [main args]"
    exit 0
fi

trainer_num=$1
output_path=$2
shift 2
if [[ $trainer_num = cuda* ]]; then
    cuda_devices=`echo $trainer_num | sed 's/cuda://'`
    trainer_num=`echo $cuda_devices | awk -F',' '{print NF}'`
else
    cuda_devices=`python script/available_gpu.py --best $trainer_num`
fi

WORKROOT=$(cd $(dirname $0); pwd)
cd $WORKROOT

#### paddle ####
# 选择要使用的GPU
export CUDA_VISIBLE_DEVICES=$cuda_devices
# CPU 核数
export CPU_NUM=$trainer_num
#### python ####
export PYTHONPATH=$WORKROOT:$WORKROOT/third:$WORKROOT/third/ERNIE:$PYTHONPATH
echo "PYTHONPATH=$PYTHONPATH"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
[ -d $output_path ] || mkdir -p $output_path
[ -f $output_path/train.log ] && mv $output_path/train.log $output_path/train.log.`date +%Y%m%d_%H%M%S`
echo "running command: ($PYTHON_BIN $@ --output $output_path)" > $output_path/train.log
python ./script/text2sql_main.py $@ --mode train --output $output_path 2>&1 | tee -a $output_path/train.log
exit $?

