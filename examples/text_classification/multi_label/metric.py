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

import paddle
from paddle.metric import Metric

class F1Score(Metric):
    """
    F1-score is the harmonic mean of precision and recall. Micro-averaging is 
    to create a global confusion matrix for all examples, and then calculate 
    the F1-score. This class is using to evaluate the performance of Dialogue 
    Slot Filling.
    """

    def __init__(self, name='F1Score', *args, **kwargs):
        super(F1Score, self).__init__(*args, **kwargs)
        self._name = name
        self.reset()

    def reset(self):
        """
        Resets all of the metric state.
        """
        self.tp = {}
        self.fn = {}
        self.fp = {}

    def update(self, probs, labels):
        """
        Update the states based on the current mini-batch prediction results.
        Args:
            probs (Tensor): The predicted value is a Tensor with 
                shape [batch_size, num_labels] and type float32 or 
                float64.
            labels (Tensor): The ground truth value is a 2D Tensor, 
                its shape is [batch_size, num_labels] and type is int64.
        """
        probs = logits.numpy()
        labels = labels.numpy()
        assert probs.shape[0] == labels.shape[0]
        assert probs.shape[1] == labels.shape[1]
        for i in range(logits.shape[0]):
            start, end = 1, probs.shape[1]
            while end > start:
                if labels[i][end - 1] != 0:
                    break
                end -= 1
            prob, label = probs[i][start:end], labels[i][start:end]
            for y_pred, y in zip(prob, label):
                if y_pred == y:
                    self.tp[y] = self.tp.get(y, 0) + 1
                else:
                    self.fp[y_pred] = self.fp.get(y_pred, 0) + 1
                    self.fn[y] = self.fn.get(y, 0) + 1

    def accumulate(self):
        """
        Calculate the final micro F1 score.
        Returns:
            A scaler float: results of the calculated micro F1 score.
        """
        tp_total = sum(self.tp.values())
        fn_total = sum(self.fn.values())
        fp_total = sum(self.fp.values())
        p_total = float(tp_total) / (tp_total + fp_total)
        r_total = float(tp_total) / (tp_total + fn_total)
        if p_total + r_total == 0:
            return 0
        f1_micro = 2 * p_total * r_total / (p_total + r_total)
        return f1_micro

    def name(self):
        """
        Returns metric name
        """
        return self._name
