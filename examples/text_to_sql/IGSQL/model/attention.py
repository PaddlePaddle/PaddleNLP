#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""Contains classes for computing and keeping track of attention distributions.
"""
from collections import namedtuple

import paddle
import paddle.nn.functional as F
import math
import numpy as np

np.random.seed(0)


class AttentionResult(
        namedtuple('AttentionResult', ('scores', 'distribution', 'vector'))):
    """Stores the result of an attention calculation."""
    __slots__ = ()


class Attention(paddle.nn.Layer):
    """Attention mechanism class. Stores parameters for and computes attention.

    Attributes:
       transform_query (`bool`): Whether or not to transform the query being
           passed in with a weight transformation before computing attentino.
       transform_key (`bool`): Whether or not to transform the key being
           passed in with a weight transformation before computing attentino.
       transform_value (`bool`): Whether or not to transform the value being
           passed in with a weight transformation before computing attentino.
       key_size (`int`): The size of the key vectors.
       value_size (`int`): The size of the value vectors.
           the query or key.
       query_weights (`Parameter`): Weights for transforming the query.
       key_weights (`Parameter`): Weights for transforming the key.
       value_weights (`Parameter`): Weights for transforming the value.
    """

    def __init__(self, query_size, key_size, value_size):
        super().__init__()
        self.key_size = key_size
        self.value_size = value_size

        _initializer = paddle.nn.initializer.XavierUniform()

        query_weights = paddle.ParamAttr(initializer=_initializer)

        self.query_linear = paddle.nn.Linear(query_size,
                                             self.key_size,
                                             weight_attr=query_weights,
                                             bias_attr=False)

    def transform_arguments(self, query, keys, values):
        """ Transforms the query/key/value inputs before attention calculations.

        Arguments:
            query (`Tensor`): Vector representing the query (e.g., hidden state.)
            keys (`list`): List of vectors representing the key
                values.
            values (`list`): List of vectors representing the values.

        Returns:
            `triple`: The first represents the (transformed)
                query, the second represents the (transformed and concatenated)
                keys, and the third represents the (transformed and concatenated)
                values.
        """
        assert len(keys) == len(values)

        all_keys = paddle.stack(keys, axis=1)
        all_values = paddle.stack(values, axis=1)

        assert all_keys.shape[
            0] == self.key_size, "Expected key size of " + str(
                self.key_size) + " but got " + str(all_keys.shape[0])
        assert all_values.shape[0] == self.value_size

        if query.dim() == 1:
            query = query.unsqueeze(0)
        query = self.query_linear(query)

        return query, all_keys, all_values

    def forward(self, query, keys, values=None):
        if not values:
            values = keys

        query_t, keys_t, values_t = self.transform_arguments(
            query, keys, values)

        scores = paddle.t(paddle.mm(query_t, keys_t))  # len(key) x len(query)

        distribution = F.softmax(scores, axis=0)  # len(key) x len(query)

        context_vector = paddle.mm(
            values_t, distribution).squeeze()  # value_size x len(query)

        return AttentionResult(scores, distribution, context_vector)
