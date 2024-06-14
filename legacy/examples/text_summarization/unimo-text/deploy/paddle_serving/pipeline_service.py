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

from paddlenlp.data import Pad
from paddlenlp.ops.ext_utils import load
from paddlenlp.transformers import UNIMOTokenizer
from paddlenlp.utils.log import logger


def convert_example(example, tokenizer, max_seq_len=512, return_length=True):
    """Convert all examples into necessary features."""
    source = example
    tokenized_example = tokenizer.gen_encode(
        source,
        max_seq_len=max_seq_len,
        add_start_token_for_decoding=True,
        return_length=True,
        is_split_into_words=False,
    )
    return tokenized_example


def batchify_fn(batch_examples, pad_val, pad_right=False):
    """Batchify a batch of examples."""

    def pad_mask(batch_attention_mask):
        """Pad attention_mask."""
        batch_size = len(batch_attention_mask)
        max_len = max(map(len, batch_attention_mask))
        attention_mask = np.ones((batch_size, max_len, max_len), dtype="float32") * -1e9
        for i, mask_data in enumerate(attention_mask):
            seq_len = len(batch_attention_mask[i])
            if pad_right:
                mask_data[:seq_len:, :seq_len] = np.array(batch_attention_mask[i], dtype="float32")
            else:
                mask_data[-seq_len:, -seq_len:] = np.array(batch_attention_mask[i], dtype="float32")
        # In order to ensure the correct broadcasting mechanism, expand one
        # dimension to the second dimension (n_head of Transformer).
        attention_mask = np.expand_dims(attention_mask, axis=1)
        return attention_mask

    pad_func = Pad(pad_val=pad_val, pad_right=pad_right, dtype="int32")
    input_ids = pad_func([example["input_ids"] for example in batch_examples])
    token_type_ids = pad_func([example["token_type_ids"] for example in batch_examples])
    attention_mask = pad_mask([example["attention_mask"] for example in batch_examples])
    seq_len = np.asarray([example["seq_len"] for example in batch_examples], dtype="int32")
    input_dict = {}
    input_dict["input_ids"] = input_ids
    input_dict["token_type_ids"] = token_type_ids
    input_dict["attention_mask"] = attention_mask
    input_dict["seq_len"] = seq_len
    return input_dict


def postprocess_response(token_ids, tokenizer):
    """Post-process the decoded sequence. Truncate from the first <eos>."""
    eos_pos = len(token_ids)
    for i, tok_id in enumerate(token_ids):
        if tok_id == tokenizer.mask_token_id:
            eos_pos = i
            break
    token_ids = token_ids[:eos_pos]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = tokenizer.merge_subword(tokens)
    return tokens


class UnimoTextOp(Op):
    """Op for unimo_text."""

    def init_op(self):
        self.tokenizer = UNIMOTokenizer.from_pretrained("unimo-text-1.0-summary")

    def preprocess(self, input_dicts, data_id, log_id):
        # Convert input format
        ((_, input_dict),) = input_dicts.items()
        data = input_dict["inputs"]
        if isinstance(data, str) and "array(" in data:
            data = eval(data)
        else:
            logger.error("input value  {}is not supported.".format(data))
        data = [i.decode("utf-8") for i in data]
        examples = [convert_example(i, self.tokenizer) for i in data]
        input_dict = batchify_fn(examples, self.tokenizer.pad_token_id)
        # the first return must be a dict or a list of dict, the dict corresponding to a batch of model input
        return input_dict, False, None, ""

    def postprocess(self, input_dicts, fetch_dict, data_id, log_id):
        outputs = fetch_dict["transpose_0.tmp_0"]
        results = []
        for sample in outputs:
            result = []
            for idx, beam in enumerate(sample):
                if idx >= len(sample) // 2:
                    break
                res = "".join(postprocess_response(beam, self.tokenizer))
                result.append(res)
            results.append(result)
        out_dict = {}
        out_dict["outputs"] = str(results)
        # the first return must be a dict or a list of dict, the dict corresponding to a batch of model output
        return out_dict, None, ""


class UnimoTextService(WebService):
    def get_pipeline_response(self, read_op):
        return UnimoTextOp(name="text_summarization", input_ops=[read_op])


if __name__ == "__main__":
    # Load FastGeneration lib.
    load("FastGeneration", verbose=True)
    service = UnimoTextService(name="text_summarization")
    service.prepare_pipeline_config("config.yml")
    service.run_service()
