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

en_to_cn = {
    "bod": "身体",
    "mic": "微生物类",
    "dis": "疾病",
    "sym": "临床表现",
    "pro": "医疗程序",
    "equ": "医疗设备",
    "dru": "药物",
    "dep": "科室",
    "ite": "医学检验项目",
}

label_list = [
    [
        "B-bod",
        "I-bod",
        "E-bod",
        "S-bod",
        "B-dis",
        "I-dis",
        "E-dis",
        "S-dis",
        "B-pro",
        "I-pro",
        "E-pro",
        "S-pro",
        "B-dru",
        "I-dru",
        "E-dru",
        "S-dru",
        "B-ite",
        "I-ite",
        "E-ite",
        "S-ite",
        "B-mic",
        "I-mic",
        "E-mic",
        "S-mic",
        "B-equ",
        "I-equ",
        "E-equ",
        "S-equ",
        "B-dep",
        "I-dep",
        "E-dep",
        "S-dep",
        "O",
    ],
    ["B-sym", "I-sym", "E-sym", "S-sym", "O"],
]


def _extract_chunk(tokens):
    chunks = set()
    start_idx, cur_idx = 0, 0
    while cur_idx < len(tokens):
        if tokens[cur_idx][0] == "B":
            start_idx = cur_idx
            cur_idx += 1
            while cur_idx < len(tokens) and tokens[cur_idx][0] == "I":
                if tokens[cur_idx][2:] == tokens[start_idx][2:]:
                    cur_idx += 1
                else:
                    break
            if cur_idx < len(tokens) and tokens[cur_idx][0] == "E":
                if tokens[cur_idx][2:] == tokens[start_idx][2:]:
                    chunks.add((tokens[cur_idx][2:], start_idx - 1, cur_idx))
                    cur_idx += 1
        elif tokens[cur_idx][0] == "S":
            chunks.add((tokens[cur_idx][2:], cur_idx - 1, cur_idx))
            cur_idx += 1
        else:
            cur_idx += 1
    return list(chunks)


class NERPostHandler(BasePostHandler):
    def __init__(self):
        super().__init__()

    @classmethod
    def process(cls, data, parameters):
        if "logits" not in data or "logits_1" not in data:
            raise ValueError(
                "The output of model handler do not include the 'logits', "
                " please check the model handler output. The model handler output:\n{}".format(data)
            )
        tokens_oth = np.array(data["logits"])
        tokens_sym = np.array(data["logits_1"])
        tokens_oth = np.argmax(tokens_oth, axis=-1)
        tokens_sym = np.argmax(tokens_sym, axis=-1)
        entity = []
        for oth_ids, sym_ids in zip(tokens_oth, tokens_sym):
            token_oth = [label_list[0][x] for x in oth_ids]
            token_sym = [label_list[1][x] for x in sym_ids]
            chunks = _extract_chunk(token_oth) + _extract_chunk(token_sym)
            sub_entity = []
            for etype, sid, eid in chunks:
                sub_entity.append({"type": en_to_cn[etype], "start_id": sid, "end_id": eid})
            entity.append(sub_entity)
        return {"entity": entity}


app = SimpleServer()
app.register(
    "models/cblue_ner",
    model_path="../../../export_ner",
    tokenizer_name="ernie-health-chinese",
    model_handler=TokenClsModelHandler,
    post_handler=NERPostHandler,
)
