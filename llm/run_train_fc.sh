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

export PYTHONPATH=/root/paddlejob/workspace/wujingjing/models/projects/github/PaddleNLP-json-mode/:$PYTHONPATH

# convert tool-alpaca & convert & train
rm -rf ./checkpoints/qwen_lora_ckpts
python -m function_call.data_converters.convert_tool_alpaca \
        /root/paddlejob/workspace/wujingjing/models/projects/llm_data/tool_alpaca/train.json \
        ./data/tool_alpaca/messages/train.json 

python -m qwen.convert_fc_train ./data/tool_alpaca/messages/train.json ./data/tool_alpaca/train.json
cp ./data/tool_alpaca/train.json ./data/tool_alpaca/dev.json

python finetune_generation.py ./qwen/lora_1b8_argument.json
