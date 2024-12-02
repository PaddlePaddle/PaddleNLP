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

import json

import requests

# 示例文本和维度参数
texts = ["Hello, world!", "This is a test."]
dimension = 768  # 假设维度为768，根据实际情况调整

# 构建请求数据
data = {"texts": texts, "dimension": dimension}

# 将请求数据转换为JSON格式
json_data = json.dumps(data)

# 目标API的URL
url = "http://127.0.0.1:8000/v1/EMPTY"  # 假设API运行在本地的8000端口，根据实际情况调整

# 发送POST请求
response = requests.post(url, data=json_data, headers={"Content-Type": "application/json"})

# 解析响应
if response.status_code == 200:
    result = response.json()
    if "result" in result and result["result"] is not None:
        embeddings = result["result"]
        print("Embeddings:", embeddings)
    else:
        print("Error:", result)
else:
    print("Failed to get response from API. Status code:", response.status_code)
    if "error_code" in response.json():
        error_info = response.json()
        print("Error code:", error_info["error_code"])
        print("Error message:", error_info["error_msg"])
