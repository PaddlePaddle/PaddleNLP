# coding:utf-8
# Copyright (c) 2023  PaddlePaddle Authors. All Rights Reserved.
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


class CustomModelHandler(BaseModelHandler):
    def __init__(self):
        super().__init__()

    @classmethod
    def process(cls, predictor, tokenizer, data, parameters):
        max_seq_len = 128
        batch_size = 1
        if "max_seq_len" not in parameters:
            max_seq_len = parameters["max_seq_len"]
        if "batch_size" not in parameters:
            batch_size = parameters["batch_size"]
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
        examples = []
        for idx, data in enumerate(text):
            if has_pair:
                result = tokenizer(text=text[idx], text_pair=text_pair[idx], max_length=max_seq_len)
            else:
                result = tokenizer(text=text[idx], max_length=max_seq_len)
            examples.append((result["input_ids"], result["token_type_ids"]))

        # Separates data into some batches.
        batches = [examples[i : i + batch_size] for i in range(0, len(examples), batch_size)]

        def batchify_fn(samples):
            return Tuple(
                Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),
                Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),
            )(samples)

        results = [[]] * predictor._output_num
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
                predictor._predictor.run(None, {"input_ids": input_ids, "token_type_ids": token_type_ids})
                for i, out in enumerate(output):
                    results[i].append(out)

        # Resolve the logits result and get the predict label and confidence
        results_concat = []
        for i in range(0, len(results)):
            results_concat.append(np.concatenate(results[i], axis=0))
        out_dict = {"logits": results_concat[0].tolist(), "data": data}
        for i in range(1, len(results_concat)):
            out_dict[f"logits_{i}"] = results_concat[i].tolist()
        return out_dict


class ERNIEMHandler(BaseModelHandler):
    def __init__(self):
        super().__init__()

    @classmethod
    def process(cls, predictor, tokenizer, data, parameters):
        max_seq_len = 128
        batch_size = 1
        if "max_seq_len" not in parameters:
            max_seq_len = parameters["max_seq_len"]
        if "batch_size" not in parameters:
            batch_size = parameters["batch_size"]
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
        examples = []
        for idx, data in enumerate(text):
            if has_pair:
                result = tokenizer(text=text[idx], text_pair=text_pair[idx], max_length=max_seq_len)
            else:
                result = tokenizer(text=text[idx], max_length=max_seq_len)
            examples.append(result["input_ids"])

        # Separates data into some batches.
        batches = [examples[i : i + batch_size] for i in range(0, len(examples), batch_size)]

        def batchify_fn(samples):
            return Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64")(samples)

        results = [[]] * predictor._output_num
        for batch in batches:
            input_ids = batchify_fn(batch)
            if predictor._predictor_type == "paddle_inference":
                predictor._input_handles[0].copy_from_cpu(input_ids)
                predictor._predictor.run()
                output = [output_handle.copy_to_cpu() for output_handle in predictor._output_handles]
                for i, out in enumerate(output):
                    results[i].append(out)
            else:
                predictor._predictor.run(None, {"input_ids": input_ids})
                for i, out in enumerate(output):
                    results[i].append(out)

        # Resolve the logits result and get the predict label and confidence
        results_concat = []
        for i in range(0, len(results)):
            results_concat.append(np.concatenate(results[i], axis=0))
        out_dict = {"logits": results_concat[0].tolist(), "data": data}
        for i in range(1, len(results_concat)):
            out_dict[f"logits_{i}"] = results_concat[i].tolist()
        return out_dict
