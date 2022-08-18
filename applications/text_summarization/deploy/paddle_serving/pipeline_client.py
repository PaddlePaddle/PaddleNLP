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
import time
import numpy as np


class Runner(object):

    def __init__(
        self,
        server_url: str,
    ):
        self.client = PipelineClient()
        self.client.connect([server_url])

    def Run(self, data):
        inputs = np.array([i.encode('utf-8') for i in data], dtype=np.object_)
        start_time = time.time()
        ret = self.client.predict(feed_dict={"inputs": inputs})
        end_time = time.time()
        print("time cost :{} seconds".format(end_time - start_time))
        if not ret.value:
            print('Fail to fetch summary.')
        # ret is special class but a dict
        for d, s in zip(data, eval(ret.value[0])):
            print("input text: ", d)
            print("inferenced summary: ", s)
            print("--------------------")
        return


if __name__ == "__main__":
    server_url = "127.0.0.1:18011"
    runner = Runner(server_url)
    texts = [
        "深度学习是人工智能的核心技术领域。",
        "黑苦荞茶的功效与作用及食用方法",
        "百度飞桨作为中国首个自主研发、",
        "百度飞桨作为中国首个自主研发、",
        "黑苦荞茶的功效与作用及食用方法",
    ]
    # texts = ["深","学","习","是","人"]
    # texts = ['百度飞桨作为中国首个自主研发、功能丰富、开源开放的产业级深度学习平台,将从多层次技术产品、产业AI人才培养和强大的生态资源支持三方面全面护航企业实现快速AI转型升级。']
    runner.Run(texts)
