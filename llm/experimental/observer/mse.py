# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# import numpy as np
import paddle
from paddle.nn.quant.format import LinearDequanter, LinearQuanter
from paddle.quantization.factory import ObserverFactory
from paddleslim.quant.observers.uniform import UniformObserver


class MSEObserver(ObserverFactory):
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
        super(MSEObserver, self).__init__(quant_bits=quant_bits)

    def _get_class(self):
        return MSEObserverLayer


class MSEObserverLayer(UniformObserver):
    def __init__(self, layer, quant_bits=8, moving_avg=False):
        super(MSEObserverLayer, self).__init__(quant_bits=quant_bits)
        self.quant_bits = quant_bits
        self.calibration_loss = float("inf")
        self.qmin, self.qmax = self.qmin_qmax

        self._current_iters = 0
        self._range_update_factor_min = 0.001

        self._moving_avg = moving_avg
        self._max = None
        self.observer_enabled = True

    def forward(self, inputs):
        """Calculate forward pass."""
        self._scale = None
        self._zero_point = None
        self._min = None
        self._max = None

        if self.observer_enabled:
            self._max = self.cal_abs_max(inputs)

        return inputs

    def cal_abs_max(self, inputs):
        self._current_iters += 1
        # abs_max_value = float(paddle.max(paddle.abs(inputs.flatten())))
        abs_max_value = paddle.max(paddle.mean(paddle.abs(inputs.flatten()), axis=0))  # average over batch
        abs_max_value = 1e-8 if abs_max_value == 0.0 else abs_max_value
        s = 0.3
        scale_mse = abs_max_value
        while s <= 1.0:
            scale = s * abs_max_value
            s += 0.02

            quant_var_func = LinearQuanter(scale, 0.0, bit_length=self.quant_bits)
            dequant_var_func = LinearDequanter(scale, 0.0, bit_length=self.quant_bits)

            quant_var = quant_var_func(inputs)
            quant_dequant_var = dequant_var_func(quant_var)

            mse_loss = ((inputs - quant_dequant_var) ** 2).mean()
            if mse_loss <= self.calibration_loss:
                self.calibration_loss = mse_loss
                scale_mse = scale

        # import pdb;pdb.set_trace()
        if self._moving_avg and self._max is not None:
            update_factor = 1.0 / self._current_iters
            update_factor = max(update_factor, self._range_update_factor_min)
            scale_mse = self._max * (1 - update_factor) + scale_mse * update_factor

        return scale_mse

    def cal_thresholds(self):
        """Compute thresholds for MAX function."""
        if self._scale is not None:
            self._zero_point = 0
            return
        self._scale = self._max
        self._zero_point = 0

    def min_value(self) -> float:
        return 0

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
