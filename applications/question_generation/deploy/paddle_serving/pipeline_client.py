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
        inputs = data
        start_time = time.time()
        ret = self.client.predict(feed_dict={"inputs": inputs})
        end_time = time.time()
        print("time cost :{} seconds".format(end_time - start_time))
        if not ret.value:
            print('Fail to fetch summary.')
        # ret is special class but a dict
        for d, s in zip(data, eval(ret.value[0])):
            print("--------------------")
            print("input: ", d)
            print("output: ", s)
            print("--------------------")
        return


if __name__ == "__main__":
    server_url = "127.0.0.1:18011"
    runner = Runner(server_url)
    requests = [{
        "context":
        "平安银行95511电话按9转报案人工服务。 1.寿险 :95511转1 2.信用卡 95511转2 3.平安银行 95511转3 4.一账通 95511转4转8 5.产险 95511转5 6.养老险团体险 95511转6 7.健康险 95511转7 8.证券 95511转8 9.车险报案95511转9 0.重听",
        "answer": "95511"
    }]
    runner.Run(requests)
