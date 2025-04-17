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
        data = np.array([x.encode("utf-8") for x in data], dtype=np.object_)
        ret = self.client.predict(feed_dict={"sentence": data})
        for (
            d,
            l,
        ) in zip(data, eval(ret.value[0])):
            print("text: ", d)
            label = ",".join([label_list[int(ll)] for ll in l.split(",")])
            print("label: ", label)
            print("--------------------")
        return


if __name__ == "__main__":
    server_url = "127.0.0.1:18090"
    runner = Runner(server_url)
    text = ["消失的“外企光环”，5月份在华裁员900余人，香饽饽变“臭”了？", "卡车超载致使跨桥侧翻，没那么简单", "金属卡扣安装不到位，上海乐扣乐扣贸易有限公司将召回捣碎器1162件"]
    label_list = [
        "交往",
        "交往##会见",
        "交往##感谢",
        "交往##探班",
        "交往##点赞",
        "交往##道歉",
        "产品行为",
        "产品行为##上映",
        "产品行为##下架",
        "产品行为##发布",
        "产品行为##召回",
        "产品行为##获奖",
        "人生",
        "人生##产子/女",
        "人生##出轨",
        "人生##分手",
        "人生##失联",
        "人生##婚礼",
        "人生##庆生",
        "人生##怀孕",
        "人生##死亡",
        "人生##求婚",
        "人生##离婚",
        "人生##结婚",
        "人生##订婚",
        "司法行为",
        "司法行为##举报",
        "司法行为##入狱",
        "司法行为##开庭",
        "司法行为##拘捕",
        "司法行为##立案",
        "司法行为##约谈",
        "司法行为##罚款",
        "司法行为##起诉",
        "灾害/意外",
        "灾害/意外##地震",
        "灾害/意外##坍/垮塌",
        "灾害/意外##坠机",
        "灾害/意外##洪灾",
        "灾害/意外##爆炸",
        "灾害/意外##袭击",
        "灾害/意外##起火",
        "灾害/意外##车祸",
        "竞赛行为",
        "竞赛行为##夺冠",
        "竞赛行为##晋级",
        "竞赛行为##禁赛",
        "竞赛行为##胜负",
        "竞赛行为##退役",
        "竞赛行为##退赛",
        "组织关系",
        "组织关系##停职",
        "组织关系##加盟",
        "组织关系##裁员",
        "组织关系##解散",
        "组织关系##解约",
        "组织关系##解雇",
        "组织关系##辞/离职",
        "组织关系##退出",
        "组织行为",
        "组织行为##开幕",
        "组织行为##游行",
        "组织行为##罢工",
        "组织行为##闭幕",
        "财经/交易",
        "财经/交易##上市",
        "财经/交易##出售/收购",
        "财经/交易##加息",
        "财经/交易##涨价",
        "财经/交易##涨停",
        "财经/交易##融资",
        "财经/交易##跌停",
        "财经/交易##降价",
        "财经/交易##降息",
    ]
    runner.Run(text, label_list)
