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

import requests

url = "http://0.0.0.0:8189/taskflow/uie"

headers = {"Content-Type": "application/json"}
texts = [
    "威尔哥（Virgo）减速炸弹是由瑞典FFV军械公司专门为瑞典皇家空军的攻击机实施低空高速轰炸而研制，1956年开始研制，1963年进入服役，装备于A32“矛盾”、A35“龙”、和AJ134“雷”攻击机，主要用于攻击登陆>艇、停放的飞机、高炮、野战火炮、轻型防护装甲车辆以及有生力量。"
]

data = {"data": {"text": texts}}
r = requests.post(url=url, headers=headers, data=json.dumps(data))
datas = json.loads(r.text)
print(datas)
