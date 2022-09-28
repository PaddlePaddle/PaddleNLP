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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

from models.conv import GraphSageConv, ErnieSageV2Conv


class Encoder(nn.Layer):
    """ Base class 
    Chose different type ErnieSage class.
    """

    def __init__(self, config):
        """init function

        Args:
            config (Dict): all configs.
        """
        super(Encoder, self).__init__()
        self.config = config
        # Don't add ernie to self, oterwise, there will be more copies of ernie weights
        # self.ernie = ernie

    @classmethod
    def factory(cls, config, ernie):
        """Classmethod for ernie sage model.

        Args:
            config (Dict): all configs.
            ernie (nn.Layer): the ernie model.

        Raises:
            ValueError: Invalid ernie sage model type.

        Returns:
            Class: real model class.
        """
        model_type = config.model_type
        if model_type == "ErnieSageV2":
            return ErnieSageV2Encoder(config, ernie)
        else:
            raise ValueError("Invalid ernie sage model type")

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class ErnieSageV2Encoder(Encoder):

    def __init__(self, config, ernie):
        """ Ernie sage v2 encoder

        Args:
            config (Dict): all config.
            ernie (nn.Layer): the ernie model.
        """
        super(ErnieSageV2Encoder, self).__init__(config)
        # Don't add ernie to self, oterwise, there will be more copies of ernie weights
        # self.ernie = ernie
        self.convs = nn.LayerList()
        initializer = None
        fc_lr = self.config.lr / 0.001
        erniesage_conv = ErnieSageV2Conv(ernie,
                                         ernie.config["hidden_size"],
                                         self.config.hidden_size,
                                         learning_rate=fc_lr,
                                         cls_token_id=self.config.cls_token_id,
                                         aggr_func="sum")
        self.convs.append(erniesage_conv)
        for i in range(1, self.config.num_layers):
            layer = GraphSageConv(self.config.hidden_size,
                                  self.config.hidden_size,
                                  learning_rate=fc_lr,
                                  aggr_func="sum")
            self.convs.append(layer)

        if self.config.final_fc:
            self.linear = nn.Linear(
                self.config.hidden_size,
                self.config.hidden_size,
                weight_attr=paddle.ParamAttr(learning_rate=fc_lr))

    def take_final_feature(self, feature, index):
        """Gather the final feature.

        Args:
            feature (Tensor): the total featue tensor.
            index (Tensor): the index to gather.

        Returns:
            Tensor: final result tensor.
        """
        feat = paddle.gather(feature, index)
        if self.config.final_fc:
            feat = self.linear(feat)
        if self.config.final_l2_norm:
            feat = F.normalize(feat, axis=1)
        return feat

    def forward(self, graphs, term_ids, inputs):
        """ forward train function of the model.

        Args:
            graphs (Graph List): list of graph tensors.
            inputs (Tensor List): list of input tensors.

        Returns:
            Tensor List: list of final feature tensors.
        """
        # term_ids for ErnieSageConv is the raw feature.
        feature = term_ids
        for i in range(len(graphs), self.config.num_layers):
            graphs.append(graphs[0])
        for i in range(0, self.config.num_layers):
            if i == self.config.num_layers - 1 and i != 0:
                act = None
            else:
                act = "leaky_relu"
            feature = self.convs[i](graphs[i], feature, act)

        final_feats = [self.take_final_feature(feature, x) for x in inputs]
        return final_feats
