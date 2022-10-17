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
import argparse

from paddle_serving_server.web_service import WebService, Op

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', default="rocketqa-zh-base-query-encoder", help="Select tokenizer name to for model")
args = parser.parse_args()
# yapf: enable


def convert_example(example,
                    tokenizer,
                    max_seq_length=512,
                    pad_to_max_seq_len=False):
    result = []
    for text in example:
        encoded_inputs = tokenizer(text=text,
                                   max_seq_len=max_seq_length,
                                   pad_to_max_seq_len=pad_to_max_seq_len)
        input_ids = encoded_inputs["input_ids"]
        token_type_ids = encoded_inputs["token_type_ids"]
        result += [input_ids, token_type_ids]
    return result


class ErnieService(WebService):

    def init_service(self):
        from paddlenlp.transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    def preprocess(self, feed=[], fetch=[]):
        from paddlenlp.data import Stack, Tuple, Pad
        print("input dict", feed)
        batch_size = len(feed)
        is_batch = True
        examples = []
        for i in range(batch_size):
            input_ids, segment_ids = convert_example([feed[i]], self.tokenizer)
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
        return feed_dict, fetch, is_batch

    def postprocess(self, feed=[], fetch=[], fetch_map=None):
        for key in fetch_map:
            fetch_map[key] = fetch_map[key].tolist()
        return fetch_map


if __name__ == "__main__":
    ernie_service = ErnieService(name="ernie")
    ernie_service.load_model_config("../../serving_server")
    ernie_service.prepare_server(workdir="workdir", port=8080)
    ernie_service.init_service()
    ernie_service.run_debugger_service()
    ernie_service.run_web_service()
