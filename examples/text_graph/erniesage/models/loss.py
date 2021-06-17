# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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

import os

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def LossFactory(config):
    """Choose different type of loss by config

    Args:
        config (Dict): config file.

    Raises:
        ValueError: invalid loss type.

    Returns:
        Class: the real class object.
    """
    loss_type = config.loss_type
    if loss_type == "hinge":
        return HingeLoss(config.margin)
    elif loss_type == "softmax_with_cross_entropy":
        return SoftmaxWithCrossEntropy()
    else:
        raise ValueError("invalid loss type")


class SoftmaxWithCrossEntropy(nn.Layer):
    """ softmax with cross entropy loss
    """

    def __init__(self, config):
        super(SoftmaxWithCrossEntropy, self).__init__()

    def forward(self, logits, label):
        return F.cross_entropy(logits, label, reduction="mean")


class HingeLoss(nn.Layer):
    """ Hinge Loss for the pos and neg.
    """

    def __init__(self, margin):
        super(HingeLoss, self).__init__()
        self.margin = margin

    def forward(self, pos, neg):
        """ forward function

        Args:
            pos (Tensor): pos score.
            neg (Tensor): neg score.

        Returns:
            Tensor: final hinge loss.
        """
        loss = paddle.mean(F.relu(neg - pos + self.margin))
        return loss
