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
from paddle.quantization.factory import ObserverFactory

from .uniform import UniformObserver


class AbsmaxObserver(ObserverFactory):
    r"""
    It collects maximum absolute values of target tensor.
    Args:
        bit_length(int, optional): Number of bits to represent an quantized integer in binary.
        dtype(str, optional): The data type of input tensor.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.
    Examples:
       .. code-block:: python
            from paddle.quantization import QuantConfig
            from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver
            quanter = FakeQuanterWithAbsMaxObserver(moving_rate=0.99)
            q_config = QuantConfig(activation=quanter, weight=quanter)
    """

    def __init__(self, quant_bits=8):
        super(AbsmaxObserver, self).__init__(quant_bits=quant_bits)

    def _get_class(self):
        return AbsmaxObserverLayer


class AbsmaxObserverLayer(UniformObserver):
    def __init__(
        self,
        layer,
        quant_bits=8,
    ):
        super(AbsmaxObserverLayer, self).__init__(quant_bits=quant_bits)
        self._quant_bits = quant_bits
        self._layer = layer
        self._scale = None
        self._zero_point = None
        self._min = None
        self._max = paddle.to_tensor(1e-7, dtype="float32")
        self.step = 0

    def forward(self, inputs):
        """Calculate forward pass."""
        self._min, self._max = self.cal_min_max(inputs)
        return inputs

    def cal_min_max(self, inputs):
        abs_max_val = paddle.max(paddle.abs(inputs.cast("float32")))
        abs_max_val = paddle.maximum(abs_max_val, self._max)
        return 0, abs_max_val

    def cal_thresholds(self):
        """Compute thresholds for MAX function."""
        if self._scale is not None:
            self._zero_point = 0
            return
        self._scale, self._zero_point = self.cal_scales_zero_points()

    def min_value(self) -> float:
        return self._min

    def max_value(self) -> float:
        return self._max

    def bit_length(self):
        """Return the bit length of quantized data."""
        return self._quant_bits

    def quant_axis(self):
        """Return quantization axis."""
        return -1

    def scales(self):
        """Return output scales."""
        if self._scale is None:
            self.cal_thresholds()
        return self._scale

    def zero_points(self):
        """Return output zero points."""
        if self._zero_point is None:
            self.cal_thresholds()
        return self._zero_point
