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

_LOGGER = logging.getLogger()


class ErnieSeqClsOp(Op):

    def init_op(self):
        from paddlenlp.transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-medium-zh",
                                                       use_faster=True)
        # Output nodes may differ from model to model
        # You can see the output node name in the conf.prototxt file of serving_server
        self.fetch_names = [
            "linear_113.tmp_1",
        ]

    def preprocess(self, input_dicts, data_id, log_id):
        # convert input format
        (_, input_dict), = input_dicts.items()
        data = input_dict["sentence"]
        if isinstance(data, str) and "array(" in data:
            data = eval(data)
        else:
            _LOGGER.error("input value  {}is not supported.".format(data))
        data = [i.decode('utf-8') for i in data]

        # tokenizer + pad
        data = self.tokenizer(data,
                              max_length=128,
                              padding=True,
                              truncation=True)
        input_ids = data["input_ids"]
        token_type_ids = data["token_type_ids"]
        # print("input_ids:", input_ids)
        # print("token_type_ids", token_type_ids)
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
        result = fetch_dict[self.fetch_names[0]]
        max_value = np.max(result, axis=1, keepdims=True)
        exp_data = np.exp(result - max_value)
        probs = exp_data / np.sum(exp_data, axis=1, keepdims=True)
        out_dict = {
            "label": result.argmax(axis=-1),
            "confidence": probs.max(axis=-1)
        }
        return out_dict, None, ""


class ErnieSeqClsService(WebService):

    def get_pipeline_response(self, read_op):
        return ErnieSeqClsOp(name="seq_cls", input_ops=[read_op])


if __name__ == "__main__":
    ocr_service = ErnieSeqClsService(name="seq_cls")
    ocr_service.prepare_pipeline_config("seq_cls_config.yml")
    ocr_service.run_service()
