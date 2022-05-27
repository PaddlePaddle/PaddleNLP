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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

__all__ = ['RDropLoss']

class RDropLoss(nn.Layer):
    """
    R-Drop Loss implementation
    For more information about R-drop please refer to this paper: https://arxiv.org/abs/2106.14448
    Original implementation please refer to this code: https://github.com/dropreg/R-Drop
    """
    def __init__(self, reduction='none'):
        """
        reduction(obj:`str`, optional): Indicate how to average the loss,
        the candicates are ``'none'`` | ``'batchmean'`` | ``'mean'`` | ``'sum'``.
        If `reduction` is ``'mean'``, the reduced mean loss is returned;
        If `reduction` is ``'batchmean'``, the sum loss divided by batch size is returned;
        if `reduction` is ``'sum'``, the reduced sum loss is returned;
        if `reduction` is ``'none'``, no reduction will be apllied.
        Default is ``'none'``.
        """
        super(RDropLoss, self).__init__()
        if reduction not in ['sum', 'mean', 'none', 'batchmean']:
            raise ValueError(
                "'reduction' in 'RDropLoss' should be 'sum', 'mean' 'batchmean', or 'none', "
                "but received {}.".format(reduction))
        self.reduction = reduction

    def forward(self, p, q, pad_mask=None):
        """
        Args:
            p(obj:`Tensor`): the first forward logits of training examples.
            q(obj:`Tensor`): the second forward logits of training examples.
            pad_mask(obj:`Tensor`, optional): The Tensor containing the binary mask to index with, it's data type is bool.

        Returns:
            loss(obj:`Tensor`): the rdrop loss of p and q
        """
        p_loss = F.kl_div(F.log_softmax(p, axis=-1), F.softmax(q, axis=-1), reduction=self.reduction)
        q_loss = F.kl_div(F.log_softmax(q, axis=-1), F.softmax(p, axis=-1), reduction=self.reduction)

        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss = paddle.masked_select(p_loss, pad_mask)
            q_loss = paddle.masked_select(q_loss, pad_mask)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()
        loss = (p_loss + q_loss) / 2
        return loss
