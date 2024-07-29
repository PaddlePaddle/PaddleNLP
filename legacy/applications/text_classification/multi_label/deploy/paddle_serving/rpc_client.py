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
from paddle_serving_server.pipeline import PipelineClient


class Runner(object):
    def __init__(
        self,
        server_url: str,
    ):
        self.client = PipelineClient()
        self.client.connect([server_url])

    def Run(self, data, label_list):
        sentence = np.array([x.encode("utf-8") for x in data], dtype=np.object_)
        ret = self.client.predict(feed_dict={"sentence": sentence})
        for d, l in zip(data, eval(ret.value[0])):
            print("data: ", d)
            label = ",".join([label_list[int(ll)] for ll in l.split(",")])
            print("label: ", label)
            print("--------------------")
        return


if __name__ == "__main__":
    server_url = "127.0.0.1:18090"
    runner = Runner(server_url)
    text = [
        "五松新村房屋是被告婚前购买的；",
        "被告于2016年3月将车牌号为皖B×××××出售了2.7万元，被告通过原告偿还了齐荷花人民币2.6万元，原、被告尚欠齐荷花2万元。",
        "2、判令被告返还借婚姻索取的现金33万元，婚前个人存款10万元；",
        "一、判决原告于某某与被告杨某某离婚；",
    ]
    label_list = [
        "婚后有子女",
        "限制行为能力子女抚养",
        "有夫妻共同财产",
        "支付抚养费",
        "不动产分割",
        "婚后分居",
        "二次起诉离婚",
        "按月给付抚养费",
        "准予离婚",
        "有夫妻共同债务",
        "婚前个人财产",
        "法定离婚",
        "不履行家庭义务",
        "存在非婚生子",
        "适当帮助",
        "不履行离婚协议",
        "损害赔偿",
        "感情不和分居满二年",
        "子女随非抚养权人生活",
        "婚后个人财产",
    ]
    runner.Run(text, label_list)
