#!/bin/bash

# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

./${DEMO_NAME} -batch_size 8 -device ${DEVICE} -gpu_id 0 -model_dir ${MODEL_FILE_DIR} -vocab_file ${VOCAB_DIR} -data_file ${DATA_DIR}
