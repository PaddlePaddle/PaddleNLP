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

    def Run(self, text):
        sentence = np.array([t.encode('utf-8') for t in text], dtype=np.object_)
        sentence = sentence.__repr__()
        data = {"key": ["sentence"], "value": [sentence]}
        data = json.dumps(data)

        ret = requests.post(url=self.server_url, data=data)
        ret = ret.json()
        for t, l in zip(text, eval(ret['value'][0])):
            print("text: ", t)
            print("label: ", l)
            print("--------------------")
        return


if __name__ == "__main__":
    server_url = "http://127.0.0.1:5594/seq_cls/prediction"
    runner = Runner(server_url)
    text = [
        "五松新村房屋是被告婚前购买的；",
        "被告于2016年3月将车牌号为皖B×××××出售了2.7万元，被告通过原告偿还了齐荷花人民币2.6万元，原、被告尚欠齐荷花2万元。",
        "2、判令被告返还借婚姻索取的现金33万元，婚前个人存款10万元；", "一、判决原告于某某与被告杨某某离婚；"
    ]
    runner.Run(text)
