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

#0.环境准备:安装numactl
# apt-get update
# apt-get install numactl

# 1. download XFT
if [ ! -d xFasterTransformer ]; then
    git clone https://github.com/intel/xFasterTransformer.git
fi

#2.cp patch
cd xFasterTransformer
git reset --hard 420a493f5c3c74f5fdd786f5399aacd04e021df7
cd ..

if lscpu | grep -q "avx512_bf16"; then
    echo "apply bf16 and fp16."
    if [ ! -f 0001-fp16_bf16.patch ]; then
        echo "Error:  0001-fp16_bf16.patch not exist."
        exit 1
    fi
    # apply patch
    cp ./0001-fp16_bf16.patch  ./xFasterTransformer/paddle.patch
else
    echo "apply fp32 "
    if [ ! -f 0001-fp32.patch ]; then
        echo "Error:  does 0001-fp32.patch not exist."
        exit 1
    fi
    cp ./0001-fp32.patch  ./xFasterTransformer/paddle.patch
fi

#3. apply patch
cd xFasterTransformer
git apply paddle.patch

# #4. build xFasterTransformer
sh ./3rdparty/prepare_oneccl.sh
source ./3rdparty/oneccl/build/_install/env/setvars.sh

rm -rf build
mkdir build && cd build
cmake ..
make -j
cd ..

#xft
export XFT_HEADER_DIR=$PWD
export XFT_LIB_DIR=$XFT_HEADER_DIR/build
export LD_LIBRARY_PATH=$XFT_LIB_DIR:$LD_LIBRARY_PATH
#setup cpu paddle_nlp ops
cd ..
python ./src/setup_cpu.py install --user
