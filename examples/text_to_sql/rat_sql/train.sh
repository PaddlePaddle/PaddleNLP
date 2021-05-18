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

#### gpu libs ####
# 添加cuda, cudnn库的路径
export LD_LIBRARY_PATH=/home/work/cuda-10.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/work/cuda-10.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH
# 单独添加库的路径
##export LD_LIBRARY_PATH=/home/work/cudnn/cudnn_v7.4/cuda/lib64:$LD_LIBRARY_PATH
# 添加NCCL库的路径。单卡训练时非必须
export LD_LIBRARY_PATH=/home/work/nccl_2.3.5/lib/:$LD_LIBRARY_PATH

#### paddle ####
# 是否是分布式训练，0标识是分布式，1标识是单机
export PADDLE_IS_LOCAL=1
# 申请显存比例
export FLAGS_fraction_of_gpu_memory_to_use=0.9
# 选择要使用的GPU
export CUDA_VISIBLE_DEVICES=$cuda_devices
# CPU 核数
export CPU_NUM=$trainer_num
# 表示是否使用垃圾回收策略来优化网络的内存使用，<0表示禁用，>=0表示启用
export FLAGS_eager_delete_tensor_gb=0
# 是否使用快速垃圾回收策略
export FLAGS_fast_eager_deletion_mode=0
# 垃圾回收策略释放变量的内存大小百分比，范围为[0.0, 1.0]
export FLAGS_memory_fraction_of_eager_deletion=0.3
# 如果为1，则会在allreduce_op_handle中调用cudaStreamSynchronize（nccl_stream），这种模式在某些情况下可以获得更好的性能
#export FLAGS_sync_nccl_allreduce=1

#### python ####
export PYTHONPATH=$WORKROOT:$WORKROOT/third:$WORKROOT/third/ERNIE:$PYTHONPATH
echo "PYTHONPATH=$PYTHONPATH"
## python 3.6/3.7 is recomended
PYTHON_BIN=`which python3`

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
[ -d $output_path ] || mkdir -p $output_path
[ -f $output_path/train.log ] && mv $output_path/train.log $output_path/train.log.`date +%Y%m%d_%H%M%S`
echo "running command: ($PYTHON_BIN $@ --output $output_path)" > $output_path/train.log
$PYTHON_BIN -u ./script/text2sql_main.py $@ --mode train --output $output_path 2>&1 | tee -a $output_path/train.log
exit $?

