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


import json
import os

import numpy as np
from paddle.metric import Metric
from paddle.nn import BCEWithLogitsLoss
from sklearn.metrics import classification_report, f1_score

from paddlenlp.utils.log import logger


def read_local_dataset(data_path, data_file=None, shuffle_choices=False, is_test=False):
    """
    Load datasets with one example per line, formated as:
        {"text_a": X, "text_b": X, "question": X, "choices": [A, B], "labels": [0, 1]}
    """
    if data_file is not None:
        file_paths = [os.path.join(data_path, fname) for fname in os.listdir(data_path) if fname.endswith(data_file)]
    else:
        file_paths = [data_path]
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as fp:
            for example in fp:
                example = json.loads(example.strip())
                if (
                    len(" ".join(example["choices"])) + len(example["question"]) + 6 >= 2400
                    or len(example["choices"]) < 2
                ):
                    logger.warning("Skip example: " + json.dumps(example, ensure_ascii=False))
                    continue
                if "text_b" not in example:
                    example["text_b"] = ""
                if not is_test:
                    if not isinstance(example["labels"], list):
                        example["labels"] = [example["labels"]]
                    one_hots = np.zeros(len(example["choices"]), dtype="float32")
                    for x in example["labels"]:
                        one_hots[x] = 1
                    example["labels"] = one_hots.tolist()

                if shuffle_choices:
                    rand_index = np.random.permutation(len(example["choices"]))
                    example["choices"] = [example["choices"][index] for index in rand_index]
                    if not is_test:
                        example["labels"] = [example["labels"][index] for index in rand_index]
                std_keys = ["text_a", "text_b", "question", "choices", "labels"]
                std_example = {k: example[k] for k in std_keys}
                yield std_example


class BCEWithLogitsLossPaddedWithMinusOne(BCEWithLogitsLoss):
    def __init__(self, weight=None, reduction="mean", pos_weight=None, name=None):
        super(BCEWithLogitsLossPaddedWithMinusOne, self).__init__(
            weight=weight, reduction=reduction, pos_weight=pos_weight, name=name
        )

    def forward(self, logit, label):
        logit = logit[label != -100]
        label = label[label != -100]
        return super(BCEWithLogitsLossPaddedWithMinusOne, self).forward(logit, label)


class MetricReport(Metric):
    """
    F1 score for multi-label text classification task.
    """

    def __init__(self, name="MetricReport", average="micro"):
        super(MetricReport, self).__init__()
        self.average = average
        self._name = name
        self.reset()

    def reset(self):
        """
        Resets all of the metric state.
        """
        self.y_prob = None
        self.y_true = None

    def f1_score(self, y_prob):
        """
        Compute micro f1 score and macro f1 score
        """
        threshold = 0.5
        self.y_pred = y_prob > threshold
        micro_f1_score = f1_score(y_pred=self.y_pred, y_true=self.y_true, average="micro")
        macro_f1_score = f1_score(y_pred=self.y_pred, y_true=self.y_true, average="macro")
        return micro_f1_score, macro_f1_score

    def update(self, probs, labels):
        """
        Update the probability and label
        """
        if self.y_prob is not None:
            self.y_prob = np.append(self.y_prob, probs.numpy(), axis=0)
        else:
            self.y_prob = probs.numpy()
        if self.y_true is not None:
            self.y_true = np.append(self.y_true, labels.numpy(), axis=0)
        else:
            self.y_true = labels.numpy()

    def accumulate(self):
        """
        Returns micro f1 score and macro f1 score
        """
        micro_f1_score, macro_f1_score = self.f1_score(y_prob=self.y_prob)
        return micro_f1_score, macro_f1_score

    def report(self):
        """
        Returns classification report
        """
        self.y_pred = self.y_prob > 0.5
        logger.info("classification report:\n" + classification_report(self.y_true, self.y_pred, digits=4))

    def name(self):
        """
        Returns metric name
        """
        return self._name
