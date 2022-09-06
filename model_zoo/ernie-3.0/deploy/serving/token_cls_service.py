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
from paddle_serving_server.web_service import WebService, Op

from numpy import array

import logging
import numpy as np
import json

_LOGGER = logging.getLogger()


class ErnieTokenClsOp(Op):

    def init_op(self):
        from paddlenlp.transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-medium-zh",
                                                       use_faster=True)
        # The label names of NER models trained by different data sets may be different
        self.label_names = [
            'O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC'
        ]
        # Output nodes may differ from model to model
        # You can see the output node name in the conf.prototxt file of serving_server
        self.fetch_names = [
            "linear_113.tmp_1",
        ]

    def get_input_data(self, input_dicts):
        (_, input_dict), = input_dicts.items()
        data = input_dict["tokens"]

        if isinstance(data, str) and "array(" in data:
            data = eval(data)
            data = data.tolist()
        else:
            _LOGGER.error("input value {} is not supported.".format(data))
        return data

    def preprocess(self, input_dicts, data_id, log_id):
        """
        Args:
            input_dicts: input data to be preprocessed
            data_id: inner unique id, increase auto
            log_id: global unique id for RTT, 0 default
        Return:
            output_data: data for process stage
            is_skip_process: skip process stage or not, False default
            prod_errcode: None default, otherwise, product errores occured.
                          It is handled in the same way as exception. 
            prod_errinfo: "" default
        """
        # convert input format
        data = self.get_input_data(input_dicts)

        # tokenizer + pad
        is_split_into_words = False
        if isinstance(data[0], list):
            is_split_into_words = True
        data = self.tokenizer(data,
                              max_length=128,
                              padding=True,
                              truncation=True,
                              is_split_into_words=is_split_into_words)

        input_ids = data["input_ids"]
        token_type_ids = data["token_type_ids"]
        return {
            "input_ids": np.array(input_ids, dtype="int64"),
            "token_type_ids": np.array(token_type_ids, dtype="int64")
        }, False, None, ""

    def postprocess(self, input_dicts, fetch_dict, data_id, log_id):
        """
        In postprocess stage, assemble data for next op or output.
        Args:
            input_data: data returned in preprocess stage, dict(for single predict) or list(for batch predict)
            fetch_data: data returned in process stage, dict(for single predict) or list(for batch predict)
            data_id: inner unique id, increase auto
            log_id: logid, 0 default
        Returns: 
            fetch_dict: fetch result must be dict type.
            prod_errcode: None default, otherwise, product errores occured.
                          It is handled in the same way as exception.
            prod_errinfo: "" default
        """
        input_data = self.get_input_data(input_dicts)
        result = fetch_dict[self.fetch_names[0]]
        tokens_label = result.argmax(axis=-1).tolist()
        # 获取batch中每个token的实体
        value = []
        for batch, token_label in enumerate(tokens_label):
            start = -1
            label_name = ""
            items = []
            for i, label in enumerate(token_label):
                if (self.label_names[label] == "O"
                        or "B-" in self.label_names[label]) and start >= 0:
                    entity = input_data[batch][start:i - 1]
                    if isinstance(entity, list):
                        entity = "".join(entity)
                    items.append({
                        "pos": [start, i - 2],
                        "entity": entity,
                        "label": label_name,
                    })
                    start = -1
                if "B-" in self.label_names[label]:
                    start = i - 1
                    label_name = self.label_names[label][2:]
            if start >= 0:
                items.append({
                    "pos": [start, len(token_label) - 1],
                    "entity":
                    input_data[batch][start:len(token_label) - 1],
                    "label":
                    label_name,
                })
            value.append(items)
        out_dict = {
            "value": json.dumps(value),
            "tokens_label": json.dumps(tokens_label)
        }
        # print(out_dict)
        return out_dict, None, ""


class ErnieTokenClsService(WebService):

    def get_pipeline_response(self, read_op):
        return ErnieTokenClsOp(name="token_cls", input_ops=[read_op])


if __name__ == "__main__":
    ocr_service = ErnieTokenClsService(name="token_cls")
    ocr_service.prepare_pipeline_config("token_cls_config.yml")
    ocr_service.run_service()
