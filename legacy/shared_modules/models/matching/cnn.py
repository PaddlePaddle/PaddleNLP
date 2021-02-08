#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""
cnn class
"""

import paddle_layers as layers


class CNN(object):
    """
    CNN
    """

    def __init__(self, conf_dict):
        """
        initialize
        """
        self.dict_size = conf_dict["dict_size"]
        self.task_mode = conf_dict["task_mode"]
        self.emb_dim = conf_dict["net"]["emb_dim"]
        self.filter_size = conf_dict["net"]["filter_size"]
        self.num_filters = conf_dict["net"]["num_filters"]
        self.hidden_dim = conf_dict["net"]["hidden_dim"]

    def predict(self, left, right):
        """
        Forward network
        """
        # embedding layer
        emb_layer = layers.EmbeddingLayer(self.dict_size, self.emb_dim, "emb")
        left_emb = emb_layer.ops(left)
        right_emb = emb_layer.ops(right)
        # Presentation context
        cnn_layer = layers.SequenceConvPoolLayer(self.filter_size,
                                                 self.num_filters, "conv")
        left_cnn = cnn_layer.ops(left_emb)
        right_cnn = cnn_layer.ops(right_emb)
        # matching layer
        if self.task_mode == "pairwise":
            fc_layer = layers.FCLayer(self.hidden_dim, None, "fc")
            left_fc = fc_layer.ops(left_cnn)
            right_fc = fc_layer.ops(right_cnn)
            cos_sim_layer = layers.CosSimLayer()
            pred = cos_sim_layer.ops(left_fc, right_fc)
            return left_fc, pred
        else:
            concat_layer = layers.ConcatLayer(1)
            concat = concat_layer.ops([left_cnn, right_cnn])
            fc_layer = layers.FCLayer(self.hidden_dim, None, "fc")
            concat_fc = fc_layer.ops(concat)
            softmax_layer = layers.FCLayer(2, "softmax", "cos_sim")
            pred = softmax_layer.ops(concat_fc)
            return left_cnn, pred
