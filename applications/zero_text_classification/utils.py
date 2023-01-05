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
import paddle
from paddle.metric import Metric
from sklearn.metrics import classification_report, f1_score

from paddlenlp.utils.log import logger


def read_local_dataset(data_path, data_file=None, is_test=False):
    """
    Load datasets with one example per line, formated as:
        {"text_a": X, "text_b": X, "question": X, "choices": [A, B], "labels": [0, 1]}
    """
    if data_file is not None:
        file_paths = [os.path.join(data_path, fname) for fname in os.listdir(data_path) if fname.endswith(data_file)]
    else:
        file_paths = [data_path]
    skip_count = 0
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as fp:
            for example in fp:
                example = json.loads(example.strip())
                if len(example["choices"]) < 2 or not isinstance(example["text_a"], str) or len(example["text_a"]) < 3:
                    skip_count += 1
                    continue
                if "text_b" not in example:
                    example["text_b"] = ""
                if not is_test or "labels" in example:
                    if not isinstance(example["labels"], list):
                        example["labels"] = [example["labels"]]
                    one_hots = np.zeros(len(example["choices"]), dtype="float32")
                    for x in example["labels"]:
                        one_hots[x] = 1
                    example["labels"] = one_hots.tolist()

                if is_test:
                    yield example
                    continue
                std_keys = ["text_a", "text_b", "question", "choices", "labels"]
                std_example = {k: example[k] for k in std_keys if k in example}
                yield std_example
    logger.warning(f"Skip {skip_count} examples.")


class UTCLoss(object):
    def __call__(self, logit, label):
        return self.forward(logit, label)

    def forward(self, logit, label):
        logit = (1.0 - 2.0 * label) * logit
        logit_neg = logit - label * 1e12
        logit_pos = logit - (1.0 - label) * 1e12
        zeros = paddle.zeros_like(logit[..., :1])
        logit_neg = paddle.concat([logit_neg, zeros], axis=-1)
        logit_pos = paddle.concat([logit_pos, zeros], axis=-1)
        label = paddle.concat([label, zeros], axis=-1)
        logit_neg[label == -100] = -1e12
        logit_pos[label == -100] = -1e12
        neg_loss = paddle.logsumexp(logit_neg, axis=-1)
        pos_loss = paddle.logsumexp(logit_pos, axis=-1)
        loss = (neg_loss + pos_loss).mean()
        return loss


class MetricReport(Metric):
    """
    F1 score for multi-label text classification task.
    """

    def __init__(self, name="MetricReport", average="micro", threshold=0.5):
        super(MetricReport, self).__init__()
        self.average = average
        self.threshold = threshold
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
        self.y_pred = y_prob > self.threshold
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
        self.y_pred = self.y_prob > self.threshold
        logger.info("classification report:\n" + classification_report(self.y_true, self.y_pred, digits=4))

    def name(self):
        """
        Returns metric name
        """
        return self._name
