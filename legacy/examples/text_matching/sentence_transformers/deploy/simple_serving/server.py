# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from scipy.special import softmax

from paddlenlp import SimpleServer
from paddlenlp.data import Pad, Tuple
from paddlenlp.server import BaseModelHandler, BasePostHandler


class TextMatchingModelHandler(BaseModelHandler):
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
        for idx, _ in enumerate(text):
            if has_pair:
                text_a = tokenizer(text=text[idx], max_length=max_seq_len)
                text_b = tokenizer(text=text_pair[idx], max_length=max_seq_len)

            examples.append((text_a["input_ids"], text_b["input_ids"]))

        # Separates data into some batches.
        batches = [examples[i : i + batch_size] for i in range(0, len(examples), batch_size)]

        def batchify_fn(samples):
            return Tuple(
                Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),
                Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),
            )(samples)

        results = [[]] * predictor._output_num
        for batch in batches:
            query_input_ids, title_input_ids = batchify_fn(batch)
            if predictor._predictor_type == "paddle_inference":
                predictor._input_handles[0].copy_from_cpu(query_input_ids)
                predictor._input_handles[1].copy_from_cpu(title_input_ids)
                predictor._predictor.run()
                output = [output_handle.copy_to_cpu() for output_handle in predictor._output_handles]
                for i, out in enumerate(output):
                    results[i].append(out)
        print(results)

        # Resolve the logits result and get the predict label and confidence
        results_concat = []
        for i in range(0, len(results)):
            results_concat.append(np.concatenate(results[i], axis=0))

        out_dict = {"logits": results_concat[0].tolist(), "data": data}

        return out_dict


class TextMatchingPostHandler(BasePostHandler):
    def __init__(self):
        super().__init__()

    @classmethod
    def process(cls, data, parameters):
        if "logits" not in data:
            raise ValueError(
                "The output of model handler do not include the 'logits', "
                " please check the model handler output. The model handler output:\n{}".format(data)
            )

        prob_limit = 0.5
        if "prob_limit" in parameters:
            prob_limit = parameters["prob_limit"]
        logits = data["logits"]
        # softmax for probs
        logits = softmax(logits, axis=-1)

        print(logits)

        labels = []
        probs = []
        for logit in logits:
            if logit[1] > prob_limit:
                labels.append(1)
            else:
                labels.append(0)
            probs.append(logit[1])

        out_dict = {"label": labels, "similarity": probs}
        return out_dict


app = SimpleServer()
app.register(
    task_name="models/text_matching",
    model_path="../../export_model",
    tokenizer_name="ernie-3.0-medium-zh",
    model_handler=TextMatchingModelHandler,
    post_handler=TextMatchingPostHandler,
    precision="fp32",
    device_id=0,
)
