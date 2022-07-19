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

import paddle
import paddle.nn as nn


class Criterion(nn.Layer):
    '''Criterion for TPLinkerPlus'''

    def __init__(self, ghm):
        self.ghm = ghm
        self.last_weights = None  # for exponential moving averaging

    def GHM(self, gradient, bins=10, beta=0.9):
        '''
        gradient_norm: gradient_norms of all examples in this batch; (batch_size, shaking_seq_len)
        '''
        avg = paddle.mean(gradient)
        std = paddle.std(gradient) + 1e-12
        gradient_norm = nn.functional.sigmoid(
            (gradient - avg) /
            std)  # normalization and pass through sigmoid to 0 ~ 1.

        min_, max_ = paddle.min(gradient_norm), paddle.max(gradient_norm)
        gradient_norm = (gradient_norm - min_) / (max_ - min_)
        gradient_norm = paddle.clip(
            gradient_norm, 0,
            0.9999999)  # ensure elements in gradient_norm != 1.

        example_sum = paddle.flatten(gradient_norm).shape[0]  # N

        # calculate weights
        current_weights = paddle.zeros(bins)
        hits_vec = paddle.zeros(bins)
        count_hits = 0  # coungradient_normof hits
        for i in range(bins):
            bar = float((i + 1) / bins)
            hits = paddle.sum((gradient_norm <= bar)) - count_hits
            count_hits += hits
            hits_vec[i] = hits.item()
            current_weights[i] = example_sum / bins / (hits.item() +
                                                       example_sum / bins)
        # EMA: exponential moving averaging

        if self.last_weights is None:
            self.last_weights = paddle.ones(bins)  # init by ones
        current_weights = self.last_weights * beta + (1 -
                                                      beta) * current_weights
        self.last_weights = current_weights

        # weights4examples: pick weights for all examples
        weight_pk_idx = (gradient_norm / (1 / bins))[:, :, None]
        weights_rp = paddle.tile(
            current_weights[None, None, :],
            repeat_times=[gradient_norm.shape[0], gradient_norm.shape[1], 1])
        weights4examples = paddle.take_along_axis(weights_rp, weight_pk_idx,
                                                  -1).squeeze(-1)
        weights4examples /= paddle.sum(weights4examples)
        return weights4examples * gradient  # return weighted gradients

    def _multilabel_categorical_crossentropy(self, y_pred, y_true):
        """
        y_pred: (batch_size, shaking_seq_len, type_size)
        y_true: (batch_size, shaking_seq_len, type_size)
        y_true and y_pred have the same shape，elements in y_true are either 0 or 1，
             1 tags positive classes，0 tags negtive classes(means tok-pair does not have this type of link).
        """
        y_pred = (1 -
                  2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
        y_pred_pos = y_pred - (
            1 - y_true) * 1e12  # mask the pred outputs of neg classes
        zeros = paddle.zeros_like(y_pred[..., :1])  # st - st
        y_pred_neg = paddle.concat([y_pred_neg, zeros], axis=-1)
        y_pred_pos = paddle.concat([y_pred_pos, zeros], axis=-1)
        neg_loss = paddle.logsumexp(y_pred_neg, axis=-1)
        pos_loss = paddle.logsumexp(y_pred_pos, axis=-1)

        if self.ghm:
            return (self.GHM(neg_loss + pos_loss, bins=1000)).sum()
        else:
            return (neg_loss + pos_loss).mean()

    def __call__(self, y_pred, y_true):
        return self._multilabel_categorical_crossentropy(y_pred, y_true)
