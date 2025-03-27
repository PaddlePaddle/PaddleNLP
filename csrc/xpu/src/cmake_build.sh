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

set -e

rm -rf build

export PATH=/opt/output/work_dir/deepseek/deps/cmake-3.26.0-linux-x86_64/bin:$PATH
# export XDNN_PATH=Paddle/build/third_party/xpu/src/extern_xpu/xdnn-ubuntu_x86_64/ # <path_to_xdnn>
# export XRE_PATH=Paddle/build/third_party/xpu/src/extern_xpu/xre-ubuntu_x86_64/  # <path_to_xre>
# export CLANG_PATH=xtdk-ubuntu_1604_x86_64 # <path_to_xtdk>
# export HOST_SYSROOT=/opt/compiler/gcc-8.2/bin/gcc # <path_to_gcc>

export XDNN_PATH=/opt/output/work_dir/deepseek/xpu_libs/xhpc/xdnn
export XRE_PATH=/opt/output/work_dir/deepseek/xpu_libs/xre
export CLANG_PATH=/opt/output/work_dir/deepseek/xpu_libs/xdnn_plugin/xtdk_output/xtdk-llvm15-ubuntu2004_x86_64
cd plugin
bash ./cmake_build.sh
cd -

unset XDNN_PATH
unset XRE_PATH
unset CLANG_PATH

export XPU_LIB=/opt/output/work_dir/deepseek/xpu_libs

python -m pip  uninstall paddlenlp_ops -y
python setup.py install
