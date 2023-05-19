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
import time

from paddle_serving_server.pipeline import PipelineClient


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
            print("Fail to fetch summary.")
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
    requests = [
        {"context": "奇峰黄山千米以上的山峰有77座，整座黄山就是一座花岗岩的峰林，自古有36大峰，36小峰，最高峰莲花峰、最险峰天都峰和观日出的最佳点光明顶构成黄山的三大主峰。", "answer": "莲花峰"}
    ]
    runner.Run(requests)
