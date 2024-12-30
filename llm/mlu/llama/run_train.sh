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

set -ex
 
# 默认已安装的paddlepaddle 已做如下修改，如若重新安装paddlepaddle需执行下面的步骤
VENV_PATH=$(pip show paddlepaddle | grep 'Location' | awk '{print $2}')
sed -i '116s/if \"gpu\" not in place:/if \"gpu\" not in place and \"mlu\" not in place:/' "$VENV_PATH/paddle/nn/functional/flash_attention.py"
 
IPS="${1:-x.x.x.x,x.x.x.x,x.x.x.x,x.x.x.x}"
 
python -u -m paddle.distributed.launch \
    --devices "0,1,2,3,4,5,6,7" \
    --ips "$IPS" \
    --log_dir "log/log_llama_13b" \
    ../../run_pretrain.py \
    config/pretrain-llama_13b-tp8pp1sd4_stage1.json 2>&1 | tee log_llama_4nodes