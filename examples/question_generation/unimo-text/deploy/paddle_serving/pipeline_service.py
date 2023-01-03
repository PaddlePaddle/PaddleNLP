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

import logging

from infer_utils import batchify_fn, convert_example, postprocess_response
from paddle_serving_server.web_service import Op, WebService

from paddlenlp.ops.ext_utils import load
from paddlenlp.transformers import UNIMOTokenizer

_LOGGER = logging.getLogger(__name__)


class UnimoTextOp(Op):
    """Op for unimo_text."""

    def init_op(self):
        self.tokenizer = UNIMOTokenizer.from_pretrained("unimo-text-1.0")

    def preprocess(self, input_dicts, data_id, log_id):
        # Convert input format
        ((_, input_dict),) = input_dicts.items()
        data = input_dict["inputs"]
        if isinstance(data, str) and "array(" in data:
            data = eval(data)
        else:
            _LOGGER.error("input value  {}is not supported.".format(data))
        examples = [convert_example(i, self.tokenizer) for i in data]
        input_ids, token_type_ids, position_ids, attention_mask, seq_len = batchify_fn(
            examples, self.tokenizer.pad_token_id
        )
        new_dict = {}
        new_dict["input_ids"] = input_ids
        new_dict["token_type_ids"] = token_type_ids
        new_dict["attention_mask"] = attention_mask
        new_dict["seq_len"] = seq_len
        # the first return must be a dict or a list of dict, the dict corresponding to a batch of model input
        return new_dict, False, None, ""

    def postprocess(self, input_dicts, fetch_dict, data_id, log_id):
        # keyname refer to export_checkpoint_client/serving_client_conf.prototxt
        ids = fetch_dict["transpose_0.tmp_0"][:, 0, :].tolist()
        # scores = fetch_dict["_generated_var_3"][:, 0].tolist()

        results = ["".join(postprocess_response(sample, self.tokenizer)) for sample in ids]
        new_dict = {}
        new_dict["outputs"] = str(results)
        # the first return must be a dict or a list of dict, the dict corresponding to a batch of model output
        return new_dict, None, ""


class UnimoTextService(WebService):
    def get_pipeline_response(self, read_op):
        return UnimoTextOp(name="question_generation", input_ops=[read_op])


if __name__ == "__main__":
    # Load FastGeneration lib.
    load("FastGeneration", verbose=True)
    service = UnimoTextService(name="question_generation")
    service.prepare_pipeline_config("config.yml")
    service.run_service()
