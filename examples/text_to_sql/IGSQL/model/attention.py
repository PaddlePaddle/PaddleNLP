"""Contains classes for computing and keeping track of attention distributions.
"""
from collections import namedtuple

# import torch
# import torch.nn.functional as F

import paddle
import paddle.nn.functional as F
import math
# from . import torch_utils
import numpy as np
np.random.seed(0)


class AttentionResult(
        namedtuple('AttentionResult', ('scores', 'distribution', 'vector'))):
    """Stores the result of an attention calculation."""
    __slots__ = ()


class Attention(paddle.nn.Layer):
    """Attention mechanism class. Stores parameters for and computes attention.

    Attributes:
       transform_query (bool): Whether or not to transform the query being
           passed in with a weight transformation before computing attentino.
       transform_key (bool): Whether or not to transform the key being
           passed in with a weight transformation before computing attentino.
       transform_value (bool): Whether or not to transform the value being
           passed in with a weight transformation before computing attentino.
       key_size (int): The size of the key vectors.
       value_size (int): The size of the value vectors.
           the query or key.
       query_weights (dy.Parameters): Weights for transforming the query.
       key_weights (dy.Parameters): Weights for transforming the key.
       value_weights (dy.Parameters): Weights for transforming the value.
    """

    def __init__(self, query_size, key_size, value_size):
        super().__init__()
        self.key_size = key_size
        self.value_size = value_size

        _initializer = paddle.nn.initializer.XavierUniform()

        query_weights = paddle.ParamAttr(initializer=_initializer)

        self.query_linear = paddle.nn.Linear(
            query_size,
            self.key_size,
            weight_attr=query_weights,
            bias_attr=False)

        # 测试decoder
        # self.query_linear.weight.set_value(paddle.to_tensor(np.random.random([self.query_linear.weight.shape[0],self.query_linear.weight.shape[1]]),dtype='float32'))

    def transform_arguments(self, query, keys, values):
        """ Transforms the query/key/value inputs before attention calculations.

        Arguments:
            query (dy.Expression): Vector representing the query (e.g., hidden state.)
            keys (list of dy.Expression): List of vectors representing the key
                values.
            values (list of dy.Expression): List of vectors representing the values.

        Returns:
            triple of dy.Expression, where the first represents the (transformed)
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

        # query = torch_utils.linear_layer(query, self.query_weights)
        if query.dim() == 1:
            query = query.unsqueeze(0)
        query = self.query_linear(query)

        return query, all_keys, all_values

    def forward(self, query, keys, values=None):
        if not values:
            values = keys

        query_t, keys_t, values_t = self.transform_arguments(query, keys,
                                                             values)

        scores = paddle.t(paddle.mm(query_t, keys_t))  # len(key) x len(query)

        distribution = F.softmax(scores, axis=0)  # len(key) x len(query)

        context_vector = paddle.mm(
            values_t, distribution).squeeze()  # value_size x len(query)

        return AttentionResult(scores, distribution, context_vector)
