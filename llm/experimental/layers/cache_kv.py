# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import ParamAttr
from paddle.nn import Layer
from paddle.nn.initializer import Constant
from paddle.nn.quant.format import ConvertibleQuantedLayer


class CacheKVMatMul(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, transpose_x=False, transpose_y=False, name=None):
        return paddle.matmul(x, y, transpose_x, transpose_y, name)


class QuantizedCacheKVMatMul(ConvertibleQuantedLayer):
    def __init__(self, layer: Layer, q_config):
        super().__init__()
        # For FakeQuant
        self.activation_quanter = None
        self.weight_quanter = None
        if q_config.activation is not None:
            self.activation_quanter = q_config.activation._instance(layer)

    def forward(self, x, y, transpose_x=False, transpose_y=False, name=None):
        # qdq
        if self.activation_quanter is not None:
            y = self.activation_quanter(y)
        return paddle.matmul(x, y, transpose_x, transpose_y, name)

    def weights_to_quanters(self):
        return [("weight", "weight_quanter")]

    def activation_quanters(self):
        return ["activation_quanter"]


class ShiftSmoothCacheKVMatMul(Layer):
    """
    The computational logic of ShiftSmoothCacheKVMatMul is the same as CacheKVMatMul.
    The only difference is that its inputs are shift.
    """

    def __init__(self):
        super().__init__()
        self.sequence_parallel = False
        self.dtype = None

    def forward(
        self,
        x,
        y,
        transpose_x=False,
        transpose_y=False,
        perm_x=None,
        perm_y=None,
        use_smooth_x=False,
        use_smooth_out=False,
        name=None,
        sequence_parallel=False,
    ):
        self.sequence_parallel = sequence_parallel
        # smooth
        smooth_x, smooth_y = self._smooth(x, y, use_smooth_x)
        # transpose
        if perm_x is not None:
            smooth_x = paddle.transpose(smooth_x, perm=perm_x)
        if perm_y is not None:
            smooth_y = paddle.transpose(smooth_y, perm=perm_y)
        # matmul output
        out = paddle.matmul(smooth_x, smooth_y, transpose_x, transpose_y, name)
        if not use_smooth_out:
            return out
        else:
            # combine heads
            if self.sequence_parallel:
                out = paddle.transpose(out, perm=[2, 0, 1, 3])
            else:
                out = paddle.transpose(out, perm=[0, 2, 1, 3])
            return paddle.multiply(out, self.smooth_weight)

    def _smooth(self, x, y, use_smooth_x):
        # For ShiftSmooth
        smooth_shape = [1]
        self.dtype = y.dtype
        if not hasattr(self, "smooth_weight"):
            self.smooth_weight = self.create_parameter(
                shape=smooth_shape, attr=ParamAttr(initializer=Constant(value=1.0)), dtype=self.dtype
            )
        smooth_y = y
        smooth_y = paddle.divide(smooth_y, self.smooth_weight)

        if use_smooth_x:
            smooth_x = x
            x = paddle.multiply(smooth_x, self.smooth_weight)
        return x, smooth_y

    def convert_weight(self, smooth_weight=None):
        if smooth_weight is not None:
            self.smooth_weight.set_value(smooth_weight.squeeze().cast(self.dtype))


class QuantizedShiftSmoothCacheKVMatMul(ConvertibleQuantedLayer):
    """
    The computational logic of QuantizedShiftSmoothCacheKVMatMul is the same as RowParallelLinear.
    The only difference is that its inputs are shift.
    """

    def __init__(self, layer: Layer, q_config):
        super().__init__()

        # For FakeQuant
        self.weight_quanter = None
        self.activation_quanter = None
        self.smooth_weight = layer.smooth_weight
        if q_config.activation is not None:
            self.activation_quanter = q_config.activation._instance(layer)

    def forward(
        self,
        x,
        y,
        transpose_x=False,
        transpose_y=False,
        perm_x=None,
        perm_y=None,
        use_smooth_x=False,
        use_smooth_out=False,
        name=None,
        sequence_parallel=False,
    ):
        # smooth
        smooth_x, smooth_y = self._smooth(x, y, use_smooth_x)
        # qdq
        if self.activation_quanter is not None:
            smooth_y = self.activation_quanter(smooth_y)
        # transpose
        if perm_x is not None:
            smooth_x = paddle.transpose(smooth_x, perm=perm_x)
        if perm_y is not None:
            smooth_y = paddle.transpose(smooth_y, perm=perm_y)
        # matmul output
        out = paddle.matmul(smooth_x, smooth_y, transpose_x, transpose_y, name)
        if not use_smooth_out:
            return out
        else:
            # combine heads
            if sequence_parallel:
                out = paddle.transpose(out, perm=[2, 0, 1, 3])
            else:
                out = paddle.transpose(out, perm=[0, 2, 1, 3])
            return paddle.multiply(out, self.smooth_weight)

    def _smooth(self, x, y, use_smooth_x):
        # For ShiftSmooth
        self.dtype = y.dtype
        smooth_y = y
        smooth_y = paddle.divide(smooth_y, self.smooth_weight)

        if use_smooth_x:
            smooth_x = x
            x = paddle.multiply(smooth_x, self.smooth_weight)
        return x, smooth_y

    def weights_to_quanters(self):
        return [("weight", "weight_quanter")]

    def activation_quanters(self):
        return ["activation_quanter"]
