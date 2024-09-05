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

import numpy as np
import paddle
from paddle.quantization.factory import ObserverFactory

from .abs_max_headwise import AbsMaxHeadwiseObserverLayer


class AvgHeadwiseObserver(ObserverFactory):
    r"""
    It collects channel-wise maximum absolute values of target weights.
    Args:
        bit_length(int, optional): Number of bits to represent an quantized integer in binary.
        dtype(str, optional): The data type of input tensor.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.
    Examples:
       .. code-block:: python
            from paddle.quantization import QuantConfig
            from paddle.quantization.quanters import AbsMaxHeadwiseObserver
            quanter = AbsMaxHeadwiseObserver()
            q_config = QuantConfig(activation=None, weight=quanter)
    """

    def __init__(self, quant_bits=8, quant_axis=None, moving_avg=False):
        super(AvgHeadwiseObserver, self).__init__(quant_bits=quant_bits, quant_axis=quant_axis, moving_avg=moving_avg)

    def _get_class(self):
        return AvgHeadwiseObserverLayer


class AvgHeadwiseObserverLayer(AbsMaxHeadwiseObserverLayer):
    def __init__(self, layer, quant_bits=8, quant_axis=None, moving_avg=True):
        super(AvgHeadwiseObserverLayer, self).__init__(layer, quant_bits=quant_bits, quant_axis=quant_axis)
        self.quant_bits = quant_bits
        self._qmin, self._qmax = self.qmin_qmax
        self._max = None
        self._scale = None
        self._zero_point = None
        if quant_axis is not None:
            self._channel_axis = quant_axis
        self._current_iters = 0
        self._range_update_factor_min = 0.001
        self._moving_avg = moving_avg
        self.observer_enabled = True

    def forward(self, inputs, quant_axis=None):
        if self.observer_enabled:
            if quant_axis is not None:
                self._channel_axis = quant_axis
            self._max = self._cal_abs_max(inputs)
        return inputs

    def _cal_abs_max(self, inputs):
        self._current_iters += 1
        reduce_axis = tuple([i for i in range(len(inputs.shape)) if i != self.quant_axis()])
        abs_max_values = paddle.max(paddle.abs(inputs), axis=reduce_axis).cast("float32")
        abs_max_values = paddle.where(abs_max_values == np.float32(0.0), np.float32(1e-8), abs_max_values)
        if self._max is not None:
            if self._moving_avg:
                # exponential moving average update
                update_factor = 1.0 / self._current_iters
                update_factor = max(update_factor, self._range_update_factor_min)
                abs_max_values = self._max * (1 - update_factor) + abs_max_values * update_factor
            else:
                # normal average
                abs_max_values = (self._max * (self._current_iters - 1) + abs_max_values) / self._current_iters
        return abs_max_values

    def min_value(self) -> float:
        return 0.0

    def max_value(self) -> float:
        return self._max

    def cal_thresholds(self):
        """Compute thresholds for MAX function."""
        if self._scale is not None:
            self._zero_point = paddle.zeros_like(self._scale)
            return
        self._scale = self._max
        self._zero_point = paddle.zeros_like(self._scale)

    def scales(self):
        """Return output scales."""
        self.cal_thresholds()
        return self._scale

    def zero_points(self):
        """Return output zero points."""
        self.cal_thresholds()
        return self._zero_point
