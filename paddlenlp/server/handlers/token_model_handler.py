# coding:utf-8
# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

from ...data import Pad, Tuple
from .base_handler import BaseModelHandler


class TokenClsModelHandler(BaseModelHandler):
    def __init__(self):
        super().__init__()

    @classmethod
    def process(cls, predictor, tokenizer, data, parameters):
        max_seq_len = 128
        batch_size = 1
        return_attention_mask = False
        is_split_into_words = False
        if "max_seq_len" in parameters:
            max_seq_len = parameters["max_seq_len"]
        if "batch_size" in parameters:
            batch_size = parameters["batch_size"]
        if "return_attention_mask" in parameters:
            return_attention_mask = parameters["return_attention_mask"]
        if "is_split_into_words" in parameters:
            is_split_into_words = parameters["is_split_into_words"]
        text = None
        if "text" in data:
            text = data["text"]
        if text is None:
            return {}
        if isinstance(text, str):
            text = [text]
        has_pair = False
        if "text_pair" in data and data["text_pair"] is not None:
            text_pair = data["text_pair"]
            if isinstance(text_pair, str):
                text_pair = [text_pair]
            if len(text) != len(text_pair):
                raise ValueError("The length of text and text_pair must be same.")
            has_pair = True

        # Get the result of tokenizer
        pad = True
        if len(text) == 1:
            pad = False
        examples = []
        if has_pair:
            tokenizer_result = tokenizer(
                text=text,
                text_pair=text_pair,
                max_length=max_seq_len,
                truncation=True,
                return_attention_mask=return_attention_mask,
                is_split_into_words=is_split_into_words,
                padding=pad,
            )
        else:
            tokenizer_result = tokenizer(
                text=text,
                max_length=max_seq_len,
                truncation=True,
                return_attention_mask=return_attention_mask,
                is_split_into_words=is_split_into_words,
                padding=pad,
            )

        examples = []
        for input_ids, token_type_ids in zip(tokenizer_result["input_ids"], tokenizer_result["token_type_ids"]):
            examples.append((input_ids, token_type_ids))
        # Separates data into some batches.
        batches = [examples[i : i + batch_size] for i in range(0, len(examples), batch_size)]

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),  # segment
        ): fn(samples)
        results = [[] for i in range(0, predictor._output_num)]
        for batch in batches:
            input_ids, token_type_ids = batchify_fn(batch)
            if predictor._predictor_type == "paddle_inference":
                predictor._input_handles[0].copy_from_cpu(input_ids)
                predictor._input_handles[1].copy_from_cpu(token_type_ids)

                predictor._predictor.run()
                output = [output_handle.copy_to_cpu() for output_handle in predictor._output_handles]
                for i, out in enumerate(output):
                    results[i].append(out)
            else:
                output = predictor._predictor.run(None, {"input_ids": input_ids, "token_type_ids": token_type_ids})
                for i, out in enumerate(output):
                    results[i].append(out)

        results_concat = []
        for i in range(0, len(results)):
            results_concat.append(np.concatenate(results[i], axis=0))
        out_dict = {"logits": results_concat[0].tolist(), "data": data}
        for i in range(1, len(results_concat)):
            out_dict[f"logits_{i}"] = results_concat[i].tolist()
        if return_attention_mask:
            out_dict["attention_mask"] = tokenizer_result["attention_mask"]
        return out_dict
