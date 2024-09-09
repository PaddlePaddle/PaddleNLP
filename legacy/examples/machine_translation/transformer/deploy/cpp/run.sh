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

# Whether to use mkl or gpu 
WITH_MKL=ON
DEVICE='gpu'

# Please set:
# * Corresponding PaddlePaddle inference lib
# * Corresponding CUDA lib
# * Corresponding CUDNN lib
# * Corresponding model directory
# * Corresponding vocab directory
# * Corresponding data directory
LIB_DIR=YOUR_LIB_DIR
CUDA_LIB_DIR=YOUR_CUDA_LIB_DIR
CUDNN_LIB_DIR=YOUR_CUDNN_LIB_DIR
MODEL_DIR=YOUR_MODEL_DIR
# DATA_HOME is where paddlenlp stores dataset and can be returned by paddlenlp.utils.env.DATA_HOME.
VOCAB_DIR=DATA_HOME/WMT14ende/WMT14.en-de/wmt14_ende_data_bpe/vocab_all.bpe.33708
DATA_DIR=DATA_HOME/WMT14ende/WMT14.en-de/wmt14_ende_data_bpe/newstest2014.tok.bpe.33708.en

bash run_impl.sh ${LIB_DIR} transformer_e2e ${MODEL_DIR} ${WITH_MKL} ${DEVICE} ${CUDNN_LIB_DIR} ${CUDA_LIB_DIR} ${VOCAB_DIR} ${DATA_DIR}
