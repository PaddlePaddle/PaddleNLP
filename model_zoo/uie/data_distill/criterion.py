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
import paddle
import paddle.nn as nn


class Criterion(nn.Layer):
    '''Criterion for TPLinkerPlus'''

    def __init__(self, mask_zero=True):
        self.mask_zero = mask_zero

    def _sparse_multilabel_categorical_crossentropy(self,
                                                    y_true,
                                                    y_pred,
                                                    mask_zero=False):
        """稀疏版多标签分类的交叉熵
        说明：
            1. y_true.shape=[..., num_positive]，
            y_pred.shape=[..., num_classes]；
            2. 请保证y_pred的值域是全体实数，换言之一般情况下
            y_pred不用加激活函数，尤其是不能加sigmoid或者
            softmax；
            3. 预测阶段则输出y_pred大于0的类；
            4. 详情请看：https://kexue.fm/archives/7359 。
        """
        paddle.disable_static()
        zeros = paddle.zeros_like(y_pred[..., :1])
        y_pred = paddle.concat([y_pred, zeros], axis=-1)
        if mask_zero:
            infs = zeros + 1e12
            y_pred = paddle.concat([infs, y_pred[..., 1:]], axis=-1)
        y_pos_2 = paddle.take_along_axis(y_pred, y_true, axis=-1)
        y_pos_1 = paddle.concat([y_pos_2, zeros], axis=-1)
        if mask_zero:
            y_pred = paddle.concat([-infs, y_pred[..., 1:]], axis=-1)
            y_pos_2 = paddle.take_along_axis(y_pred, y_true, axis=-1)

        pos_loss = (-y_pos_1).exp().sum(axis=-1).log()
        all_loss = y_pred.exp().sum(axis=-1).log()
        aux_loss = y_pos_2.exp().sum(axis=-1).log() - all_loss
        aux_loss = paddle.clip(1 - paddle.exp(aux_loss), min=1e-10, max=1)
        neg_loss = all_loss + paddle.log(aux_loss)
        return pos_loss + neg_loss

    def __call__(self, y_pred, y_true):
        shape = y_pred.shape
        y_true = y_true[..., 0] * shape[2] + y_true[..., 1]
        # bs, nclass, seqlen * seqlen
        y_pred = paddle.reshape(y_pred,
                                shape=[shape[0], -1,
                                       np.prod(shape[2:])])

        loss = self._sparse_multilabel_categorical_crossentropy(
            y_true, y_pred, self.mask_zero)
        return loss.sum(axis=1).mean()
