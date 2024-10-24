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

import argparse
import logging

import numpy as np
from paddle_serving_server.web_service import Op, WebService

from paddlenlp.transformers import AutoTokenizer

_LOGGER = logging.getLogger()

FETCH_NAME_MAP = {
    "ernie-1.0-large-zh-cw": "linear_291.tmp_1",
    "ernie-3.0-xbase-zh": "linear_243.tmp_1",
    "ernie-3.0-base-zh": "linear_147.tmp_1",
    "ernie-3.0-medium-zh": "linear_75.tmp_1",
    "ernie-3.0-mini-zh": "linear_75.tmp_1",
    "ernie-3.0-micro-zh": "linear_51.tmp_1",
    "ernie-3.0-nano-zh": "linear_51.tmp_1",
    "ernie-2.0-base-en": "linear_147.tmp_1",
    "ernie-2.0-large-en": "linear_291.tmp_1",
    "ernie-m-base": "linear_147.tmp_1",
    "ernie-m-large": "linear_291.tmp_1",
}

# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument('--model_name', default="ernie-3.0-medium-zh", help="Select model to train, defaults to ernie-3.0-medium-zh.",
                    choices=["ernie-1.0-large-zh-cw", "ernie-3.0-xbase-zh", "ernie-3.0-base-zh", "ernie-3.0-medium-zh", "ernie-3.0-micro-zh", "ernie-3.0-mini-zh", "ernie-3.0-nano-zh", "ernie-2.0-base-en", "ernie-2.0-large-en", "ernie-m-base", "ernie-m-large"])
args = parser.parse_args()
# fmt: on


class Op(Op):
    def init_op(self):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        # Output nodes may differ from model to model
        # You can see the output node name in the conf.prototxt file of serving_server
        self.fetch_names = [
            FETCH_NAME_MAP[args.model_name],
        ]

    def preprocess(self, input_dicts, data_id, log_id):
        # Convert input format
        ((_, input_dict),) = input_dicts.items()
        data = input_dict["sentence"]
        if isinstance(data, str) and "array(" in data:
            data = eval(data)
        else:
            _LOGGER.error("input value  {}is not supported.".format(data))
        data = [i.decode("utf-8") for i in data]

        # tokenizer + pad
        data = self.tokenizer(
            data,
            max_length=args.max_seq_length,
            padding=True,
            truncation=True,
            return_position_ids=False,
            return_attention_mask=False,
        )
        tokenized_data = {}
        for tokenizer_key in data:
            tokenized_data[tokenizer_key] = np.array(data[tokenizer_key], dtype="int64")
        return tokenized_data, False, None, ""

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
            labels.append(",".join(label))
        return {"label": labels}, None, ""


class Service(WebService):
    def get_pipeline_response(self, read_op):
        return Op(name="seq_cls", input_ops=[read_op])


if __name__ == "__main__":
    service = Service(name="seq_cls")
    service.prepare_pipeline_config("config.yml")
    service.run_service()
