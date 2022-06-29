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
        sentence = np.array([x.encode('utf-8') for x in data], dtype=np.object_)
        ret = self.client.predict(feed_dict={"sentence": sentence})
        for d, l in zip(data, eval(ret.value[0])):
            print("data: ", d)
            print("label: ", l)
            print("--------------------")
        return


if __name__ == "__main__":
    server_url = "127.0.0.1:18090"
    runner = Runner(server_url)
    texts = [
        "经审理查明，2012年4月5日19时许，被告人王某在杭州市下城区朝晖路农贸市场门口贩卖盗版光碟、淫秽光碟时被民警当场抓获，并当场查获其贩卖的各类光碟5515张，其中5280张某属非法出版物、235张某属淫秽物品。上述事实，被告人王某在庭审中亦无异议，且有经庭审举证、质证的扣押物品清单、赃物照片、公安行政处罚决定书、抓获经过及户籍证明等书证；证人胡某、徐某的证言；出版物鉴定书、淫秽物品审查鉴定书及检查笔录等证据证实，足以认定。",
        "榆林市榆阳区人民检察院指控：2015年11月22日2时许，被告人王某某在自己经营的榆阳区长城福源招待所内，介绍并容留杨某向刘某某、白某向乔某某提供性服务各一次。",
        "静乐县人民检察院指控，2014年8月30日15时许，静乐县苏坊村村民张某某因占地问题去苏坊村半切沟静静铁路第五标施工地点阻拦施工时，遭被告人王某某阻止，张某某打电话叫来儿子李某某，李某某看到张某某躺在地上，打了王某某一耳光。于是王某某指使工人殴打李某某，致李某某受伤。经忻州市公安司法鉴定中心鉴定，李某某的损伤评定为轻伤一级。李某某被打伤后，被告人王某某为逃避法律追究，找到任某某，指使任某某作实施××的伪证，并承诺每月给1万元。同时王某某指使工人王某甲、韩某某去丰润派出所作由任某某打伤李某某的伪证，导致任某某被静乐县公安局以涉嫌××罪刑事拘留。公诉机关认为，被告人王某某的行为触犯了《中华人民共和国刑法》××、《中华人民共和国刑法》××××之规定，应以××罪和××罪追究其刑事责任，数罪并罚。"
    ]
    runner.Run(texts)
