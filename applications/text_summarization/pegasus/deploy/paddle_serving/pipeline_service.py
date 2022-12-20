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
from paddle_serving_server.web_service import Op, WebService

from paddlenlp.ops.ext_utils import load
from paddlenlp.transformers import PegasusChineseTokenizer
from paddlenlp.utils.log import logger


def convert_example(example, tokenizer, max_seq_len=512):
    """Convert all examples into necessary features."""
    tokenized_example = tokenizer(
        example, max_length=max_seq_len, padding=True, truncation=True, return_attention_mask=False
    )
    input_dict = {}
    input_dict["input_ids"] = np.asarray(tokenized_example["input_ids"], dtype="int32")
    return input_dict


class PegasusOp(Op):
    """Op for pegasus."""

    def init_op(self):
        self.tokenizer = PegasusChineseTokenizer.from_pretrained("IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese")

    def preprocess(self, input_dicts, data_id, log_id):
        # Convert input format
        ((_, input_dict),) = input_dicts.items()
        data = input_dict["inputs"]
        if isinstance(data, str) and "array(" in data:
            data = eval(data)
        else:
            logger.error("input value  {}is not supported.".format(data))
        input_dict = convert_example(list(data), self.tokenizer, max_seq_len=128)
        # the first return must be a dict or a list of dict, the dict corresponding to a batch of model input
        return input_dict, False, None, ""

    def postprocess(self, input_dicts, fetch_dict, data_id, log_id):
        outputs = fetch_dict["transpose_48.tmp_0"]
        results = []
        for sample in outputs:
            result = []
            for idx, beam in enumerate(sample):
                if idx >= len(sample) // 2:
                    break
                res = self.tokenizer.decode(beam, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                result.append(res)
            results.append(result)
        out_dict = {}
        out_dict["outputs"] = str(results)
        # the first return must be a dict or a list of dict, the dict corresponding to a batch of model output
        return out_dict, None, ""


class PegasusService(WebService):
    def get_pipeline_response(self, read_op):
        return PegasusOp(name="text_summarization", input_ops=[read_op])


if __name__ == "__main__":
    # Load FasterTransformer lib.
    load("FasterTransformer", verbose=True)
    service = PegasusService(name="text_summarization")
    service.prepare_pipeline_config("config.yml")
    service.run_service()
