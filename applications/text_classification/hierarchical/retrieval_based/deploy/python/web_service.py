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

from paddle_serving_server.web_service import Op, WebService


def convert_example(example, tokenizer, max_seq_length=512, pad_to_max_seq_len=False):
    result = []
    for text in example:
        encoded_inputs = tokenizer(
            text=text["sentence"], max_seq_len=max_seq_length, pad_to_max_seq_len=pad_to_max_seq_len
        )
        input_ids = encoded_inputs["input_ids"]
        token_type_ids = encoded_inputs["token_type_ids"]
        result += [input_ids, token_type_ids]
    return result


class ErnieOp(Op):
    def init_op(self):
        from paddlenlp.transformers import AutoTokenizer

        model_name_or_path = "rocketqa-zh-dureader-query-encoder"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def preprocess(self, input_dicts, data_id, log_id):
        from paddlenlp.data import Pad, Tuple

        ((_, input_dict),) = input_dicts.items()
        print("input dict", input_dict)
        batch_size = len(input_dict.keys())
        examples = []
        for i in range(batch_size):
            example = eval(input_dict[str(i)])
            input_ids, segment_ids = convert_example([example], self.tokenizer)
            examples.append((input_ids, segment_ids))
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id, dtype="int64"),  # input
            Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id, dtype="int64"),  # segment
        ): fn(samples)
        input_ids, segment_ids = batchify_fn(examples)
        feed_dict = {}
        feed_dict["input_ids"] = input_ids
        feed_dict["token_type_ids"] = segment_ids
        return feed_dict, False, None, ""

    def postprocess(self, input_dicts, fetch_dict, data_id, log_id):
        new_dict = {}
        new_dict["output_embedding"] = str(fetch_dict["output_embedding"].tolist())
        return new_dict, None, ""


class ErnieService(WebService):
    def get_pipeline_response(self, read_op):
        ernie_op = ErnieOp(name="ernie", input_ops=[read_op])
        return ernie_op


ernie_service = ErnieService(name="ernie")
ernie_service.prepare_pipeline_config("config_nlp.yml")
ernie_service.run_service()
