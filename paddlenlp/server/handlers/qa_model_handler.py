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

from .base_handler import BaseModelHandler


class QAModelHandler(BaseModelHandler):
    def __init__(self):
        super().__init__()

    @classmethod
    def process(cls, predictor, tokenizer, data, parameters):

        max_seq_len = 128
        doc_stride = 128
        batch_size = 1
        if "max_seq_len" in parameters:
            max_seq_len = parameters["max_seq_len"]
        if "batch_size" in parameters:
            batch_size = parameters["batch_size"]
        if "doc_stride" in parameters:
            doc_stride = parameters["doc_stride"]
        context = None
        question = None

        # Get the context in qa task
        if "context" in data:
            context = data["context"]
        if context is None:
            return {}
        if isinstance(context, str):
            context = [context]

        # Get the context in qa task
        if "question" in data:
            question = data["question"]
        if question is None:
            return {}
        if isinstance(question, str):
            question = [question]

        tokenizer_results = tokenizer(
            question,
            context,
            stride=doc_stride,
            max_length=max_seq_len,
            return_offsets_mapping=True,
            pad_to_max_seq_len=True,
        )
        input_ids = tokenizer_results["input_ids"]
        token_type_ids = tokenizer_results["token_type_ids"]
        # Separates data into some batches.
        batches = [[i, i + batch_size] for i in range(0, len(input_ids), batch_size)]

        results = [[] for i in range(0, predictor._output_num)]
        for start, end in batches:
            input_id = np.array(input_ids[start:end]).astype("int64")
            token_type_id = np.array(token_type_ids[start:end]).astype("int64")
            if predictor._predictor_type == "paddle_inference":
                predictor._input_handles[0].copy_from_cpu(input_id)
                predictor._input_handles[1].copy_from_cpu(token_type_id)

                predictor._predictor.run()
                output = [output_handle.copy_to_cpu() for output_handle in predictor._output_handles]
                for i, out in enumerate(output):
                    results[i].extend(out.tolist())
            else:
                output = predictor._predictor.run(None, {"input_ids": input_id, "token_type_ids": token_type_id})
                for i, out in enumerate(output):
                    results[i].extend(out.tolist())
        data["offset_mapping"] = tokenizer_results["offset_mapping"]
        out_dict = {"logits": results[0], "data": data}
        for i in range(1, len(results)):
            out_dict[f"logits_{i}"] = results[1]
        return out_dict
