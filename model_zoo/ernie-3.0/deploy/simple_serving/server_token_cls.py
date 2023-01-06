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


class NERPostHandler(BasePostHandler):
    def __init__(self):
        super().__init__()

    @classmethod
    def process(cls, data, parameters):
        label_list = ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "O"]
        input_datas = data["data"]["text"]
        predictions = np.array(data["logits"])
        tokens_label = predictions.argmax(axis=-1)
        tokens_label = tokens_label.tolist()
        value = []
        for batch, token_label in enumerate(tokens_label):
            start = -1
            label_name = ""
            items = []
            input_data = input_datas[batch]
            for i, label in enumerate(token_label):
                if (label_list[label] == "O" or "B-" in label_list[label]) and start >= 0:
                    entity = input_data[start : i - 1]
                    if isinstance(entity, list):
                        entity = "".join(entity)
                    items.append(
                        {
                            "pos": [start, i - 2],
                            "entity": entity,
                            "label": label_name,
                        }
                    )
                    start = -1
                if "B-" in label_list[label]:
                    start = i - 1
                    label_name = label_list[label][2:]
            if start >= 0:
                items.append(
                    {
                        "pos": [start, len(token_label) - 1],
                        "entity": input_data[start : len(token_label) - 1],
                        "label": "",
                    }
                )
            value.append(items)
        out_dict = {"value": value, "tokens_label": tokens_label}
        return out_dict


app = SimpleServer()
app.register(
    "models/ernie_ner",
    model_path="../../best_models/msra_ner/export/",
    tokenizer_name="ernie-3.0-medium-zh",
    model_handler=TokenClsModelHandler,
    post_handler=NERPostHandler,
)
