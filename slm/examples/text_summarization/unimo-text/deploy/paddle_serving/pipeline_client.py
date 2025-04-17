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

import numpy as np
from paddle_serving_server.pipeline import PipelineClient

from paddlenlp.utils.log import logger


class Runner(object):
    def __init__(
        self,
        server_url: str,
    ):
        self.client = PipelineClient()
        self.client.connect([server_url])

    def Run(self, data):
        inputs = np.array([i.encode("utf-8") for i in data], dtype=np.object_)
        start_time = time.time()
        ret = self.client.predict(feed_dict={"inputs": inputs})
        end_time = time.time()
        logger.info("time cost :{} seconds".format(end_time - start_time))
        if not ret.value:
            logger.warning("Fail to fetch summary.")
        # ret is special class but a dict
        for d, s in zip(data, eval(ret.value[0])):
            print("Text: ", d)
            print("Summary: ", s[0])
            print("-" * 50)


if __name__ == "__main__":
    server_url = "127.0.0.1:18011"
    runner = Runner(server_url)
    texts = [
        "雪后的景色可真美丽呀！不管是大树上，屋顶上，还是菜地上，都穿上了一件精美的、洁白的羽绒服。放眼望去，整个世界变成了银装素裹似的，世界就像是粉妆玉砌的一样。",
        "根据“十个工作日”原则，下轮调价窗口为8月23日24时。卓创资讯分析，原油价格或延续震荡偏弱走势，且新周期的原油变化率仍将负值开局，消息面对国内成品油市场并无提振。受此影响，预计国内成品油批发价格或整体呈现稳中下滑走势，但“金九银十”即将到来，卖方看好后期市场，预计跌幅较为有限。",
    ]
    runner.Run(texts)
