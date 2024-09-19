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

import abc
from typing import Tuple

import numpy as np
from paddle.quantization.base_observer import BaseObserver


class UniformObserver(BaseObserver):
    """This is the base class for a uniform quantization observer, which provides
    common functions for calculating the scale and zero-point used in uniform quantization.
    Uniform quantization maps floating point values to integers, where the scale determines
    the step size of the quantizer and the floating point zero is mapped to the zero-point,
    an integer value ensuring that zero is quantized without error.

    Args:
        quant_bits (int): The number of bits for quantization.
        sign (bool): Whether the quantized integer includes a sign.
        symmetric (bool): Whether it is symmetric quantization. the quantization is symmetric.
        In symmetric quantization, the range of floating point values is relaxed to be symmetric
        around zero and the zero-point is always 0.

    """

    def __init__(
        self,
        quant_bits=8,
        sign=True,
        symmetric=True,
    ):
        super(UniformObserver, self).__init__()
        self._quant_bits = quant_bits
        self._sign = sign
        self._symmetric = symmetric

        self._min = None
        self._max = None
        self._qmin = None
        self._qmax = None

        self._scale = None
        self._zero_point = None

    @property
    def qmin_qmax(self):
        """Calculate the range of the quantized integer based on the specified
        quant_bits, sign, and symmetric properties."""
        if isinstance(self._quant_bits, tuple):
            if self._quant_bits[0] == 4 and self._quant_bits[1] == 3 and len(self._quant_bits) == 2:
                self._qmin = -448.0
                self._qmax = 448.0
            elif self._quant_bits[0] == 5 and self._quant_bits[1] == 2 and len(self._quant_bits) == 2:
                self._qmin = -57344.0
                self._qmax = 57344.0
            else:
                raise NotImplementedError(
                    "Currently, only float8_e4m3 and float8_e5m2 formats are supported. Please set quant_bits to (4,3) or (5,2) for the corresponding format."
                )
        else:
            if self._sign:
                self._qmin = -(2 ** (self.bit_length() - 1))
                self._qmax = 2 ** (self.bit_length() - 1) - 1
            else:
                self._qmin = 0
                self._qmax = 2 ** self.bit_length()
        return self._qmin, self._qmax

    @abc.abstractmethod
    def min_value(self) -> float:
        """The minimum value of floating-point numbers."""
        raise NotImplementedError(
            "Please implement the abstract method to get the The minimum value of floating-point numbers."
        )

    @abc.abstractmethod
    def max_value(self) -> float:
        """The maximum value of floating-point numbers."""
        raise NotImplementedError(
            "Please implement the abstract method to get the the maximum value value of floating-point numbers."
        )

    def cal_scales_zero_points(self) -> Tuple[float, float]:
        """Calculate the scales and zero points based on the min_value and max_value."""
        assert self.min_value() is not None and self.max_value() is not None
        _qmin, _qmax = self.qmin_qmax
        # For one-sided distributions, the range (_min , _max ) is relaxed to include zero.
        # It is important to ensure that common operations like zero padding do not cause quantization errors.
        _min = min(self.min_value(), 0.0)
        _max = max(self.max_value(), 0.0)

        if self._symmetric:
            self._scale = max(-_min, _max)
            if self._sign:
                self._zero_point = 0
            else:
                self._zero_point = (_qmax + _qmin) / 2
        else:
            self._scale = (_max - _min) / float(_qmax - _qmin)
            self._zero_point = _qmin - round(_min / self._scale)
            self._zero_point = np.clip(self._zero_point, _qmin, _qmax)
        return self._scale, self._zero_point
