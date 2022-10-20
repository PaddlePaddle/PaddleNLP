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
import numpy as np
from numpy import array
import requests
import json
import sys


class Runner(object):

    def __init__(
        self,
        server_url: str,
    ):
        self.server_url = server_url

    def Run(self, text, label_list):
        sentence = np.array([t.encode('utf-8') for t in text], dtype=np.object_)
        sentence = sentence.__repr__()
        data = {"key": ["sentence"], "value": [sentence]}
        data = json.dumps(data)

        ret = requests.post(url=self.server_url, data=data)
        ret = ret.json()
        for t, l in zip(text, eval(ret['value'][0])):
            print("text: ", t)
            print("label: ", label_list[l])
            print("--------------------")
        return


if __name__ == "__main__":
    server_url = "http://127.0.0.1:9878/seq_cls/prediction"
    runner = Runner(server_url)
    text = [
        "黑苦荞茶的功效与作用及食用方法", "交界痣会凸起吗", "检查是否能怀孕挂什么科", "鱼油怎么吃咬破吃还是直接咽下去",
        "幼儿挑食的生理原因是"
    ]
    label_list = [
        '病情诊断', '治疗方案', '病因分析', '指标解读', '就医建议', '疾病表述', '后果表述', '注意事项', '功效作用',
        '医疗费用', '其他'
    ]
    runner.Run(text, label_list)
