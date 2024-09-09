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

autoconfig_json_file="$1"   # autoconfig/llama7b_pretrain_buffer.json
autoconfig_json_file_name=$(basename "$1")
model_name=${autoconfig_json_file_name%.*} 
auto_log_file=./autoconfig/${model_name}_auto_tuner.log

log="./llama7b_pretrain_auto_tuner.log"
launch_best_cfg=$(sed -n "s/.*Launch best cfg: \(.*\)}/\1/p" "$auto_log_file")
cfg_max_mem_usage=$(echo "$launch_best_cfg" | awk -F"\"max_mem_usage\":" '{print $2}' | awk -F, '{print $1}')

buffer=$(sed -n 's/.*"buffer":\([^,}]*\).*/\1/p' $autoconfig_json_file | awk '{print $1}')
max_mem_usage=$(sed -n 's/.*"max_mem_usage":\([^,}]*\).*/\1/p' $autoconfig_json_file | awk '{print $1}')
result=`expr $max_mem_usage - $buffer`

if [ $cfg_max_mem_usage -le $result ]
then
    echo "Autotuner buffer预留成功"
    exit 0
else
    echo "Autotuner buffer预留失败"
    echo "Autotuner 预设 max_mem_usgae: $max_mem_usage buffer: $buffer, 可用显存为: $result"
    echo "Autotuner best_cfg 实际使用显存为: $cfg_max_mem_usage"
    exit -1
fi