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
from sklearn.metrics import f1_score, classification_report

from paddle.metric import Metric
from paddlenlp.utils.log import logger


class MetricReport(Metric):
    """
    F1 score for hierarchical text classification task.
    """

    def __init__(self, name='MetricReport', average='micro'):
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
        micro_f1_score = f1_score(y_pred=self.y_pred,
                                  y_true=self.y_true,
                                  average='micro')
        macro_f1_score = f1_score(y_pred=self.y_pred,
                                  y_true=self.y_true,
                                  average='macro')
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
        logger.info("classification report:\n" +
                    classification_report(self.y_true, self.y_pred, digits=4))

    def name(self):
        """
        Returns metric name
        """
        return self._name
