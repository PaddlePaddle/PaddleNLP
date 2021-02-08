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
bow class
"""

import paddle_layers as layers


class BOW(object):
    """
    BOW
    """

    def __init__(self, conf_dict):
        """
        initialize
        """
        self.dict_size = conf_dict["dict_size"]
        self.task_mode = conf_dict["task_mode"]
        self.emb_dim = conf_dict["net"]["emb_dim"]
        self.bow_dim = conf_dict["net"]["bow_dim"]

    def predict(self, left, right):
        """
        Forward network
        """
        # embedding layer
        emb_layer = layers.EmbeddingLayer(self.dict_size, self.emb_dim, "emb")
        left_emb = emb_layer.ops(left)
        right_emb = emb_layer.ops(right)
        # Presentation context
        pool_layer = layers.SequencePoolLayer("sum")
        left_pool = pool_layer.ops(left_emb)
        right_pool = pool_layer.ops(right_emb)
        softsign_layer = layers.SoftsignLayer()
        left_soft = softsign_layer.ops(left_pool)
        right_soft = softsign_layer.ops(right_pool)
        # matching layer
        if self.task_mode == "pairwise":
            bow_layer = layers.FCLayer(self.bow_dim, None, "fc")
            left_bow = bow_layer.ops(left_soft)
            right_bow = bow_layer.ops(right_soft)
            cos_sim_layer = layers.CosSimLayer()
            pred = cos_sim_layer.ops(left_bow, right_bow)
            return left_bow, pred
        else:
            concat_layer = layers.ConcatLayer(1)
            concat = concat_layer.ops([left_soft, right_soft])
            bow_layer = layers.FCLayer(self.bow_dim, None, "fc")
            concat_fc = bow_layer.ops(concat)
            softmax_layer = layers.FCLayer(2, "softmax", "cos_sim")
            pred = softmax_layer.ops(concat_fc)
            return left_soft, pred
