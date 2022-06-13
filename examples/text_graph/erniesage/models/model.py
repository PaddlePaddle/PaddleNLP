# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import pgl
import paddle
import paddle.nn as nn
import numpy as np
from paddlenlp.transformers import ErniePretrainedModel

from models.encoder import Encoder
from models.loss import LossFactory

__all__ = ["ErnieSageForLinkPrediction"]


class ErnieSageForLinkPrediction(ErniePretrainedModel):
    """ErnieSage for link prediction task.
    """

    def __init__(self, ernie, config):
        """ Model which Based on the PaddleNLP PretrainedModel

        Note: 
            1. the ernie must be the first argument.
            2. must set self.XX = ernie to load weights.
            3. the self.config keyword is taken by PretrainedModel class.

        Args:
            ernie (nn.Layer): the submodule layer of ernie model. 
            config (Dict): the config file
        """
        super(ErnieSageForLinkPrediction, self).__init__()
        self.config_file = config
        self.ernie = ernie
        self.encoder = Encoder.factory(self.config_file, self.ernie)
        self.loss_func = LossFactory(self.config_file)

    def forward(self, graphs, data):
        """Forward function of link prediction task.

        Args:
            graphs (Graph List): the Graph list.
            data (Tensor List): other input of the model.

        Returns:
            Tensor: loss and output tensors.
        """
        term_ids, user_index, pos_item_index, neg_item_index, user_real_index, pos_item_real_index = data
        # encoder model
        outputs = self.encoder(graphs, term_ids,
                               [user_index, pos_item_index, neg_item_index])
        user_feat, pos_item_feat, neg_item_feat = outputs

        # calc loss
        if self.config_file.neg_type == "batch_neg":
            neg_item_feat = pos_item_feat

        pos = paddle.sum(user_feat * pos_item_feat, -1, keepdim=True)  # [B, 1]
        neg = paddle.matmul(user_feat, neg_item_feat,
                            transpose_y=True)  # [B, B]
        loss = self.loss_func(pos, neg)
        # return loss, outputs
        return loss, outputs + [user_real_index, pos_item_real_index]
