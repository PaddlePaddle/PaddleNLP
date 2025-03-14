# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

CHAT_URL = "http://127.0.0.1:8731"

data = {
    "src": [
        "Natalia sold clips to 48 of her friends in April, ",
        "Weng earns $12 an hour for babysitting. Yesterday",
    ],
    "tgt": [
        "Natalia sold 48/2 = 24 clips in May. #### 72",
        "She earned 0.2 x 50 = $10. #### 10",
    ],
    "response": [
        "Natalia sold 48+24 = 72 clips altogether in April and May. #### 72",
        "2",
    ],
}
res = requests.post(CHAT_URL, json=data)
result = json.loads(res.text)
print("result:", result, result["score"])
