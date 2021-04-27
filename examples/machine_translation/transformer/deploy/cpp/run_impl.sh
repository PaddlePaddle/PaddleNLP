#!/bin/bash
mkdir -p build
cd build
rm -rf *

LIB_DIR=$1
DEMO_NAME=$2
MODEL_FILE_DIR=$3
WITH_MKL=$4
DEVICE=$5
CUDNN_LIB=$6
CUDA_LIB=$7
VOCAB_DIR=$8
DATA_DIR=$9

WITH_GPU=OFF
if [[ $DEVICE="gpu" ]]; then
  WITH_GPU=ON
fi

cmake .. -DPADDLE_LIB=${LIB_DIR} \
  -DWITH_MKL=${WITH_MKL} \
  -DDEMO_NAME=${DEMO_NAME} \
  -DWITH_GPU=${WITH_GPU} \
  -DWITH_STATIC_LIB=OFF \
  -DUSE_TENSORRT=${USE_TENSORRT} \
  -DCUDNN_LIB=${CUDNN_LIB} \
  -DCUDA_LIB=${CUDA_LIB} \
  -DTENSORRT_ROOT=${TENSORRT_ROOT}

make -j

# <batch_size> <device> <gpu_id> <use_mkl> <threads> <model_dir> <vocab_dir> <data_dir>
./${DEMO_NAME} 8 ${DEVICE} 0 0 1 ${MODEL_FILE_DIR} ${VOCAB_DIR} ${DATA_DIR}
