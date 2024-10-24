# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

unset http_proxy && unset https_proxy
# 指定语义检索系统的Yaml配置文件
export CUDA_VISIBLE_DEVICES=0
export PIPELINE_YAML_PATH=rest_api/pipeline/text_to_image_retrieval.yaml
# 使用端口号 8891 启动模型服务
python rest_api/application.py 8891