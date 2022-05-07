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

class ErnieOp(Op):
    def init_op(self):
        from paddlenlp.transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-medium-zh")

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
        data = self.tokenizer(data, max_length=128, padding=True)
        input_ids = data["input_ids"]
        token_type_ids = data["token_type_ids"]
        # print("input_ids:", input_ids)
        # print("token_type_ids", token_type_ids)
        return {"input_ids": np.array(input_ids, dtype="int64"), "token_type_ids": np.array(token_type_ids, dtype="int64")}, False, None, ""

    def postprocess(self, input_dicts, fetch_dict, data_id, log_id):
        result = fetch_dict["linear_75.tmp_1"]
        # np.argpartition
        out_dict = {"index": result.argmax(axis=-1), "confidence": result.max(axis=-1)}
        return out_dict, None, ""

class Ernie3Service(WebService):
    def get_pipeline_response(self, read_op):
        return ErnieOp(name="ernie", input_ops=[read_op])

if __name__ == "__main__":
    ocr_service = Ernie3Service(name="erine3")
    ocr_service.prepare_pipeline_config("config.yml")
    ocr_service.run_service()

