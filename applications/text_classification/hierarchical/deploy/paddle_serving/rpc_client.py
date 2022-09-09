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
from paddle_serving_server.pipeline import PipelineClient
from numpy import array, float32

import numpy as np


class Runner(object):

    def __init__(
        self,
        server_url: str,
    ):
        self.client = PipelineClient()
        self.client.connect([server_url])

    def Run(self, data):
        data = np.array([x.encode('utf-8') for x in data], dtype=np.object_)
        ret = self.client.predict(feed_dict={"sentence": data})
        for d, l, in zip(data, eval(ret.value[0])):
            print("text: ", d)
            print("label: ", l)
            print("--------------------")
        return


if __name__ == "__main__":
    server_url = "127.0.0.1:7688"
    runner = Runner(server_url)
    texts = [
        "消失的“外企光环”，5月份在华裁员900余人，香饽饽变“臭”了？", "卡车超载致使跨桥侧翻，没那么简单",
        "金属卡扣安装不到位，上海乐扣乐扣贸易有限公司将召回捣碎器1162件"
    ]
    runner.Run(texts)
