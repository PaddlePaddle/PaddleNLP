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

cd /root/PaddleNLP/pipelines/
# linux
python utils/offline_ann.py --index_name dureader_robust_query_encoder \
                            --doc_dir data/dureader_dev \
                            --query_embedding_model rocketqa-zh-nano-query-encoder \
                            --passage_embedding_model rocketqa-zh-nano-para-encoder \
                            --port 9200 \
                            --host localhost \
                            --embedding_dim 312 \
                            --delete_index 
# 使用端口号 8891 启动模型服务
nohup python rest_api/application.py 8891 > server.log 2>&1 &
# 在指定端口 8502 启动 WebUI
nohup python -m streamlit run ui/webapp_semantic_search.py --server.port 8502 > client.log 2>&1 &
