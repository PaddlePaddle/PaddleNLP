# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from sklearn.metrics import roc_auc_score, f1_score

from paddle.metric import Metric


class MultiLabelReport(Metric):
    """
    AUC and F1Score for multi-label text classification task.
    """

    def __init__(self, name='MultiLabelReport', average='micro'):
        super(MultiLabelReport, self).__init__()
        self.average = average
        self._name = name
        self.reset()

    def f1_score(self, y_prob):
        '''
        Returns the f1 score by searching the best threshhold
        '''
        best_score = 0
        for threshold in [i * 0.01 for i in range(100)]:
            self.y_pred = y_prob > threshold
            score = f1_score(y_pred=self.y_pred,
                             y_true=self.y_true,
                             average=self.average)
            if score > best_score:
                best_score = score
        return best_score

    def reset(self):
        """
        Resets all of the metric state.
        """
        self.y_prob = None
        self.y_true = None

    def update(self, probs, labels):
        if self.y_prob is not None:
            self.y_prob = np.append(self.y_prob, probs.numpy(), axis=0)
        else:
            self.y_prob = probs.numpy()
        if self.y_true is not None:
            self.y_true = np.append(self.y_true, labels.numpy(), axis=0)
        else:
            self.y_true = labels.numpy()

    def accumulate(self):
        auc = roc_auc_score(y_score=self.y_prob,
                            y_true=self.y_true,
                            average=self.average)
        f1_score = self.f1_score(y_prob=self.y_prob)
        return auc, f1_score

    def name(self):
        """
        Returns metric name
        """
        return self._name
