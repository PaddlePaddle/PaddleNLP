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

import requests
import json

url = "http://0.0.0.0:8189/taskflow/uie"
headers = {"Content-Type": "application/json"}
texts = ['城市内交通费7月5日金额114广州至佛山', '5月9日交通费29元从北苑到望京搜后']
data = {
    'data': texts
}
r = requests.post(url=url, headers=headers, data=json.dumps(data))
datas = json.loads(r.text)
print(datas)
