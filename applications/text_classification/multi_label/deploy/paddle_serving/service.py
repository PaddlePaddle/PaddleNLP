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


class Op(Op):

    def init_op(self):
        from paddlenlp.transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-base-zh",
                                                       use_faster=True)
        # Output nodes may differ from model to model
        # You can see the output node name in the conf.prototxt file of serving_server
        self.fetch_names = [
            "linear_147.tmp_1",
        ]

    def preprocess(self, input_dicts, data_id, log_id):
        # Convert input format
        (_, input_dict), = input_dicts.items()
        data = input_dict["sentence"]
        if isinstance(data, str) and "array(" in data:
            data = eval(data)
        else:
            _LOGGER.error("input value  {}is not supported.".format(data))
        data = [i.decode('utf-8') for i in data]

        # tokenizer + pad
        data = self.tokenizer(data,
                              max_length=512,
                              padding=True,
                              truncation=True)
        input_ids = data["input_ids"]
        token_type_ids = data["token_type_ids"]
        return {
            "input_ids": np.array(input_ids, dtype="int64"),
            "token_type_ids": np.array(token_type_ids, dtype="int64")
        }, False, None, ""

    def postprocess(self, input_dicts, fetch_dict, data_id, log_id):

        results = fetch_dict[self.fetch_names[0]]
        results = np.array(results)
        labels = []

        for result in results:
            label = []
            result = 1 / (1 + (np.exp(-result)))
            for i, p in enumerate(result):
                if p > 0.5:
                    label.append(str(i))
            labels.append(','.join(label))
        return {"label": labels}, None, ""


class Service(WebService):

    def get_pipeline_response(self, read_op):
        return Op(name="seq_cls", input_ops=[read_op])


if __name__ == "__main__":
    service = Service(name="seq_cls")
    service.prepare_pipeline_config("config.yml")
    service.run_service()
