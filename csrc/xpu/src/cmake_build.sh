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

# export XDNN_PATH=Paddle/build/third_party/xpu/src/extern_xpu/xdnn-ubuntu_x86_64/ # <path_to_xdnn>
# export XRE_PATH=Paddle/build/third_party/xpu/src/extern_xpu/xre-ubuntu_x86_64/  # <path_to_xre>
# export CLANG_PATH=xtdk-ubuntu_1604_x86_64 # <path_to_xtdk>
# export HOST_SYSROOT=/opt/compiler/gcc-8.2/bin/gcc # <path_to_gcc>

cd plugin
./cmake_build.sh
cd -

python -m pip  uninstall paddlenlp_ops -y
python setup.py install
