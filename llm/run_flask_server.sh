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

model_name=__internal_testing__/tiny-random-chatglm

# flask server 测试
# model_name=__internal_testing__/tiny-fused-qwen-inference5.2
# python flask_server.py --model_name_or_path ${model_name} --port 8011 --flask_port 8012 --dtype "float32"

model_name=/root/.paddlenlp/models/qwen/qwen-7b-chat
python flask_server.py --model_name_or_path ${model_name} --port 8011 --flask_port 8012 --dtype "float16"
# python predictor.py --model_name_or_path ${model_name} --dtype "float32"
