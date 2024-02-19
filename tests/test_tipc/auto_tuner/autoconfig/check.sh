#!/usr/bin/env bash

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

autoconfig_json_file=$(basename "$1")   # autoconfig/llama7b_pretrain.json
model_name=${autoconfig_json_file%.*} 
auto_log_file=./autoconfig/${model_name}_auto_tuner.log
  
if [ -f "$auto_log_file" ] && grep -q "Launch best cfg:" "$auto_log_file"; then  
    echo "autotuner 已找到最优配置"  
    if [ -d "./autoconfig/best_cfg" ]; then  
        echo "autotuner 已执行最优配置"  
        exit 0  
    else  
        echo "autotuner 未执行最优配置"  
        exit -1  
    fi  
else  
    echo "autotuner 执行失败，请检查日志文件是否存在或是否包含指定文本！"  
    exit -1  
fi
