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
network layers
"""

import paddle.fluid as fluid
import paddle.fluid.param_attr as attr


class EmbeddingLayer(object):
    """
    Embedding Layer class
    """

    def __init__(self, dict_size, emb_dim, name="emb"):
        """
        initialize
        """
        self.dict_size = dict_size
        self.emb_dim = emb_dim
        self.name = name

    def ops(self, input):
        """
        operation
        """
        emb = fluid.embedding(
            input=input,
            size=[self.dict_size, self.emb_dim],
            is_sparse=True,
            param_attr=attr.ParamAttr(name=self.name))
        return emb


class SequencePoolLayer(object):
    """
    Sequence Pool Layer class
    """

    def __init__(self, pool_type):
        """
        initialize
        """
        self.pool_type = pool_type

    def ops(self, input):
        """
        operation
        """
        pool = fluid.layers.sequence_pool(input=input, pool_type=self.pool_type)
        return pool


class FCLayer(object):
    """
    Fully Connect Layer class
    """

    def __init__(self, fc_dim, act, name="fc"):
        """
        initialize
        """
        self.fc_dim = fc_dim
        self.act = act
        self.name = name

    def ops(self, input):
        """
        operation
        """
        fc = fluid.layers.fc(input=input,
                             size=self.fc_dim,
                             param_attr=attr.ParamAttr(name="%s.w" % self.name),
                             bias_attr=attr.ParamAttr(name="%s.b" % self.name),
                             act=self.act,
                             name=self.name)
        return fc


class DynamicGRULayer(object):
    """
    Dynamic GRU Layer class
    """

    def __init__(self, gru_dim, name="dyn_gru"):
        """
        initialize
        """
        self.gru_dim = gru_dim
        self.name = name

    def ops(self, input):
        """
        operation
        """
        proj = fluid.layers.fc(
            input=input,
            size=self.gru_dim * 3,
            param_attr=attr.ParamAttr(name="%s_fc.w" % self.name),
            bias_attr=attr.ParamAttr(name="%s_fc.b" % self.name))
        gru = fluid.layers.dynamic_gru(
            input=proj,
            size=self.gru_dim,
            param_attr=attr.ParamAttr(name="%s.w" % self.name),
            bias_attr=attr.ParamAttr(name="%s.b" % self.name))
        return gru


class DynamicLSTMLayer(object):
    """
    Dynamic LSTM Layer class
    """

    def __init__(self, lstm_dim, name="dyn_lstm"):
        """
        initialize
        """
        self.lstm_dim = lstm_dim
        self.name = name

    def ops(self, input):
        """
        operation
        """
        proj = fluid.layers.fc(
            input=input,
            size=self.lstm_dim * 4,
            param_attr=attr.ParamAttr(name="%s_fc.w" % self.name),
            bias_attr=attr.ParamAttr(name="%s_fc.b" % self.name))
        lstm, _ = fluid.layers.dynamic_lstm(
            input=proj,
            size=self.lstm_dim * 4,
            param_attr=attr.ParamAttr(name="%s.w" % self.name),
            bias_attr=attr.ParamAttr(name="%s.b" % self.name))
        return lstm


class SequenceLastStepLayer(object):
    """
    Get Last Step Sequence Layer class
    """

    def __init__(self):
        """
        initialize
        """
        pass

    def ops(self, input):
        """
        operation
        """
        last = fluid.layers.sequence_last_step(input)
        return last


class SequenceConvPoolLayer(object):
    """
    Sequence convolution and pooling Layer class
    """

    def __init__(self, filter_size, num_filters, name):
        """
        initialize
        Args: 
          filter_size:Convolution kernel size
          num_filters:Convolution kernel number
        """
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.name = name

    def ops(self, input):
        """
        operation
        """
        conv = fluid.nets.sequence_conv_pool(
            input=input,
            filter_size=self.filter_size,
            num_filters=self.num_filters,
            param_attr=attr.ParamAttr(name=self.name),
            act="relu")
        return conv


class DataLayer(object):
    """
    Data Layer class
    """

    def __init__(self):
        """
        initialize
        """
        pass

    def ops(self, name, shape, dtype, lod_level=0):
        """
        operation
        """
        data = fluid.data(
            name=name, shape=shape, dtype=dtype, lod_level=lod_level)
        return data


class ConcatLayer(object):
    """
    Connection Layer class
    """

    def __init__(self, axis):
        """
        initialize
        """
        self.axis = axis

    def ops(self, inputs):
        """
        operation
        """
        concat = fluid.layers.concat(inputs, axis=self.axis)
        return concat


class ReduceMeanLayer(object):
    """
    Reduce Mean Layer class
    """

    def __init__(self):
        """
        initialize
        """
        pass

    def ops(self, input):
        """
        operation
        """
        mean = fluid.layers.reduce_mean(input)
        return mean


class CrossEntropyLayer(object):
    """
    Cross Entropy Calculate Layer
    """

    def __init__(self, name="cross_entropy"):
        """
        initialize
        """
        pass

    def ops(self, input, label):
        """
        operation
        """
        loss = fluid.layers.cross_entropy(input=input, label=label)
        return loss


class SoftmaxWithCrossEntropyLayer(object):
    """
    Softmax with Cross Entropy Calculate Layer
    """

    def __init__(self, name="softmax_with_cross_entropy"):
        """
        initialize
        """
        pass

    def ops(self, input, label):
        """
        operation
        """
        loss = fluid.layers.softmax_with_cross_entropy(
            logits=input, label=label)
        return loss


class CosSimLayer(object):
    """
    Cos Similarly Calculate Layer
    """

    def __init__(self):
        """
        initialize
        """
        pass

    def ops(self, x, y):
        """
        operation
        """
        sim = fluid.layers.cos_sim(x, y)
        return sim


class ElementwiseMaxLayer(object):
    """
    Elementwise Max Layer class
    """

    def __init__(self):
        """
        initialize
        """
        pass

    def ops(self, x, y):
        """
        operation
        """
        max = fluid.layers.elementwise_max(x, y)
        return max


class ElementwiseAddLayer(object):
    """
    Elementwise Add Layer class
    """

    def __init__(self):
        """
        initialize
        """
        pass

    def ops(self, x, y):
        """
        operation
        """
        add = fluid.layers.elementwise_add(x, y)
        return add


class ElementwiseSubLayer(object):
    """
    Elementwise Add Layer class
    """

    def __init__(self):
        """
        initialize
        """
        pass

    def ops(self, x, y):
        """
        operation
        """
        sub = fluid.layers.elementwise_sub(x, y)
        return sub


class ConstantLayer(object):
    """
    Generate A Constant Layer class
    """

    def __init__(self):
        """
        initialize
        """
        pass

    def ops(self, input, shape, dtype, value):
        """
        operation
        """
        shape = list(shape)
        input_shape = fluid.layers.shape(input)
        shape[0] = input_shape[0]
        constant = fluid.layers.fill_constant(shape, dtype, value)
        return constant


class SigmoidLayer(object):
    """
    Sigmoid Layer class
    """

    def __init__(self):
        """
        initialize
        """
        pass

    def ops(self, input):
        """
        operation
        """
        sigmoid = fluid.layers.sigmoid(input)
        return sigmoid


class SoftsignLayer(object):
    """
    Softsign Layer class
    """

    def __init__(self):
        """
        initialize
        """
        pass

    def ops(self, input):
        """
        operation
        """
        softsign = fluid.layers.softsign(input)
        return softsign


# class MatmulLayer(object):
#     def __init__(self, transpose_x, transpose_y):
#         self.transpose_x = transpose_x
#         self.transpose_y = transpose_y

#     def ops(self, x, y):
#         matmul = fluid.layers.matmul(x, y, self.transpose_x, self.transpose_y)
#         return matmul

# class Conv2dLayer(object):
#     def __init__(self, num_filters, filter_size, act, name):
#         self.num_filters = num_filters
#         self.filter_size = filter_size
#         self.act = act
#         self.name = name

#     def ops(self, input):
#         conv = fluid.layers.conv2d(input, self.num_filters, self.filter_size, param_attr=attr.ParamAttr(name="%s.w" % self.name), bias_attr=attr.ParamAttr(name="%s.b" % self.name), act=self.act)
#         return conv

# class Pool2dLayer(object):
#     def __init__(self, pool_size, pool_type):
#         self.pool_size = pool_size
#         self.pool_type = pool_type

#     def ops(self, input):
#         pool = fluid.layers.pool2d(input, self.pool_size, self.pool_type)
#         return pool
