#!/bin/bash
set +x
set -e

work_path=$(dirname $(readlink -f $0))

# 1. check paddle_inference exists
if [ ! -d "${work_path}/lib/paddle_inference" ]; then
  echo "Please download paddle_inference lib and move it in cpp_deploy/lib"
  exit 1
fi

# 2. compile
mkdir -p build
cd build
rm -rf *

# same with the seq_cls_infer.cc
PROJECT_NAME=seq_cls_infer

WITH_MKL=ON
WITH_GPU=ON

LIB_DIR=${work_path}/lib/paddle_inference
CUDNN_LIB=/usr/lib/x86_64-linux-gnu/
CUDA_LIB=/usr/local/cuda/lib64

cmake .. -DPADDLE_LIB=${LIB_DIR} \
  -DWITH_MKL=${WITH_MKL} \
  -DPROJECT_NAME=${PROJECT_NAME} \
  -DWITH_GPU=${WITH_GPU} \
  -DWITH_STATIC_LIB=OFF \
  -DCUDNN_LIB=${CUDNN_LIB} \
  -DCUDA_LIB=${CUDA_LIB} 

make -j
