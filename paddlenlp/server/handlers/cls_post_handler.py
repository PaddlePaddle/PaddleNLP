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

from .base_handler import BasePostHandler


class MultiClassificationPostHandler(BasePostHandler):
    def __init__(self):
        super().__init__()

    @classmethod
    def process(cls, data, parameters):
        if "logits" not in data:
            raise ValueError(
                "The output of model handler do not include the 'logits', "
                " please check the model handler output. The model handler output:\n{}".format(data)
            )

        logits = data["logits"]
        logits = np.array(logits)
        max_value = np.max(logits, axis=1, keepdims=True)
        exp_data = np.exp(logits - max_value)
        probs = exp_data / np.sum(exp_data, axis=1, keepdims=True)
        out_dict = {"label": logits.argmax(axis=-1).tolist(), "confidence": probs.max(axis=-1).tolist()}
        return out_dict


class MultiLabelClassificationPostHandler(BasePostHandler):
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
        logits = np.array(logits)
        logits = 1 / (1.0 + np.exp(-logits))
        labels = []
        probs = []
        for logit in logits:
            label = []
            prob = []
            for i, p in enumerate(logit):
                if p > prob_limit:
                    label.append(i)
                    prob.append(p)
            labels.append(label)
            probs.append(prob)
        out_dict = {"label": labels, "confidence": probs}
        return out_dict
