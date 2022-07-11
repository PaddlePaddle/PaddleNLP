# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import json

from paddle_serving_server.web_service import WebService, Op


def convert_example(example, tokenizer, max_seq_length=512):

    query, title = example["query"], example["title"]
    encoded_inputs = tokenizer(text=query,
                               text_pair=title,
                               max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    return input_ids, token_type_ids


class ErnieOp(Op):

    def init_op(self):
        from paddlenlp.transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('ernie-gram-zh')

    def preprocess(self, input_dicts, data_id, log_id):
        from paddlenlp.data import Stack, Tuple, Pad

        (_, input_dict), = input_dicts.items()
        print("input dict", input_dict)
        batch_size = len(input_dict.keys())
        examples = []
        for i in range(batch_size):
            example = json.loads(input_dict[str(i)].replace("\'", "\""))
            input_ids, segment_ids = convert_example(example, self.tokenizer)
            examples.append((input_ids, segment_ids))
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id, dtype="int64"
                ),  # input
            Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id, dtype="int64"
                ),  # segment
        ): fn(samples)
        input_ids, segment_ids = batchify_fn(examples)
        feed_dict = {}
        feed_dict['input_ids'] = input_ids
        feed_dict['token_type_ids'] = segment_ids
        return feed_dict, False, None, ""

    def postprocess(self, input_dicts, fetch_dict, data_id, log_id):
        new_dict = {}
        new_dict["predict"] = str(fetch_dict["predict"].tolist())
        return new_dict, None, ""


class ErnieService(WebService):

    def get_pipeline_response(self, read_op):
        ernie_op = ErnieOp(name="ernie", input_ops=[read_op])
        return ernie_op


ernie_service = ErnieService(name="ernie")
ernie_service.prepare_pipeline_config("config_nlp.yml")
ernie_service.run_service()
