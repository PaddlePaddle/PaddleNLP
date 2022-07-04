#!/usr/bin/env bash
set -x
# add CUDA, cuDNN and NCCL to environment variable
# export LD_LIBRARY_PATH=/home/work/cuda-10.0/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# export LD_LIBRARY_PATH=/home/work/cuda-10.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/home/work/cudnn/cudnn_v7.6/cuda/lib64:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/home/work/nccl/nccl2.4.2_cuda10.1/lib:$LD_LIBRARY_PATH
export FLAGS_sync_nccl_allreduce=1
export FLAGS_fraction_of_gpu_memory_to_use=1
export FLAGS_eager_delete_tensor_gb=1.0
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_memory_fraction_of_eager_deletion=1

export iplist=`hostname -i`
unset http_proxy
unset https_proxy
set +x
