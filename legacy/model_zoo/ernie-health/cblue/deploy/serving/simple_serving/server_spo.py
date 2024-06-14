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

from paddlenlp import SimpleServer
from paddlenlp.server import BasePostHandler, TokenClsModelHandler

label_list = [
    "预防",
    "阶段",
    "就诊科室",
    "辅助治疗",
    "化疗",
    "放射治疗",
    "手术治疗",
    "实验室检查",
    "影像学检查",
    "辅助检查",
    "组织学检查",
    "内窥镜检查",
    "筛查",
    "多发群体",
    "发病率",
    "发病年龄",
    "多发地区",
    "发病性别倾向",
    "死亡率",
    "多发季节",
    "传播途径",
    "并发症",
    "病理分型",
    "相关（导致）",
    "鉴别诊断",
    "相关（转化）",
    "相关（症状）",
    "临床表现",
    "治疗后症状",
    "侵及周围组织转移的症状",
    "病因",
    "高危因素",
    "风险评估因素",
    "病史",
    "遗传因素",
    "发病机制",
    "病理生理",
    "药物治疗",
    "发病部位",
    "转移部位",
    "外侵部位",
    "预后状况",
    "预后生存率",
    "同义词",
]


class SPOPostHandler(BasePostHandler):
    def __init__(self):
        super().__init__()

    @classmethod
    def process(cls, data, parameters):
        if "logits" not in data or "logits_1" not in data:
            raise ValueError(
                "The output of model handler do not include the 'logits', "
                " please check the model handler output. The model handler output:\n{}".format(data)
            )
        lengths = np.array(data["attention_mask"], dtype="float32").sum(axis=-1)
        ent_logits = np.array(data["logits"])
        spo_logits = np.array(data["logits_1"])
        ent_pred_list = []
        ent_idxs_list = []
        for idx, ent_pred in enumerate(ent_logits):
            seq_len = lengths[idx] - 2
            start = np.where(ent_pred[:, 0] > 0.5)[0]
            end = np.where(ent_pred[:, 1] > 0.5)[0]
            ent_pred = []
            ent_idxs = {}
            for x in start:
                y = end[end >= x]
                if (x == 0) or (x > seq_len):
                    continue
                if len(y) > 0:
                    y = y[0]
                    if y > seq_len:
                        continue
                    ent_idxs[x] = (x - 1, y - 1)
                    ent_pred.append((x - 1, y - 1))
            ent_pred_list.append(ent_pred)
            ent_idxs_list.append(ent_idxs)

        spo_preds = spo_logits > 0
        spo_pred_list = [[] for _ in range(len(spo_preds))]
        idxs, preds, subs, objs = np.nonzero(spo_preds)
        for idx, p_id, s_id, o_id in zip(idxs, preds, subs, objs):
            obj = ent_idxs_list[idx].get(o_id, None)
            if obj is None:
                continue
            sub = ent_idxs_list[idx].get(s_id, None)
            if sub is None:
                continue
            spo_pred_list[idx].append((tuple(sub), p_id, tuple(obj)))
        input_data = data["data"]["text"]
        ent_list = []
        spo_list = []
        for i, (ent, rel) in enumerate(zip(ent_pred_list, spo_pred_list)):
            cur_ent_list = []
            cur_spo_list = []
            for sid, eid in ent:
                cur_ent_list.append("".join([str(d) for d in input_data[i][sid : eid + 1]]))
            for s, p, o in rel:
                cur_spo_list.append(
                    (
                        "".join([str(d) for d in input_data[i][s[0] : s[1] + 1]]),
                        label_list[p],
                        "".join([str(d) for d in input_data[i][o[0] : o[1] + 1]]),
                    )
                )
            ent_list.append(cur_ent_list)
            spo_list.append(cur_spo_list)

        return {"entity": ent_list, "spo": spo_list}


app = SimpleServer()
app.register(
    "models/cblue_spo",
    model_path="../../../export",
    tokenizer_name="ernie-health-chinese",
    model_handler=TokenClsModelHandler,
    post_handler=SPOPostHandler,
)
