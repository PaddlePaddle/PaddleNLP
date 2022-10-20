# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import base64
import time
import numpy as np
import requests

headers = {"Content-type": "application/json"}
url = "http://10.21.226.175:8080/ernie/prediction"  # XXX取决于服务端YourService的初始化name参数

data = {"feed": ["买了社保，是不是就不用买商业保险了？"], "fetch": ["output_embedding"]}
data = json.dumps(data)
print(data)
r = requests.post(url=url, headers=headers, data=data)
print(r.json())
json_data = r.json()
data = np.array(json_data['result']['output_embedding'])
print(data.shape)