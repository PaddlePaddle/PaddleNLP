"""
Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import numpy as np
import paddle
from .channel_wise import ChannelWiseObserver
from paddle.quantization.factory import ObserverFactory


class AsymCacheKVObserver(ObserverFactory):
    r"""
    It collects channel-wise or head-wise scale values of target weights.
    Args:
        bit_length(int, optional): Number of bits to represent an quantized integer in binary.
        dtype(str, optional): The data type of input tensor.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.
    Examples:
       .. code-block:: python
            from paddle.quantization import QuantConfig
            from paddle.quantization.quanters import AsymCacheKVObserver
            quanter = AsymCacheKVObserver()
            q_config = QuantConfig(activation=None, weight=quanter)
    """

    def __init__(self, quant_bits=8, quant_axis=-1, symmetric=False, moving_avg=False):
        """
            Args:
            quant_bits (int, optional): number of bits to use for quantization. Defaults to 8.
            symmetric (bool, optional): whether the quantization should be symmetric or not. Defaults to False.
        """
        super(AsymCacheKVObserver, self).__init__(
            quant_bits=quant_bits, quant_axis=quant_axis, symmetric=symmetric, moving_avg=moving_avg)

    def _get_class(self):
        """
            获取缓存层类，返回值为AsymCacheKVObserverLayer。
        如果已经初始化过，则直接返回之前的结果；否则，创建一个新的实例并返回。
        
        Returns:
            AsymCacheKVObserverLayer (type): 缓存层类，用于观察和操作缓存数据。
        """
        return AsymCacheKVObserverLayer


class AsymCacheKVObserverLayer(ChannelWiseObserver):
    """
    class: AsymCacheKVObserverLayer
    """
    def __init__(self, layer, quant_bits=8, symmetric=False, quant_axis=-1, moving_avg=False):
        """
            Args:
            layer (Layer): Layer to be observed and quantized.
            quant_bits (int, optional): Number of bits for quantization. Defaults to 8.
            symmetric (bool, optional): Whether the quantization is symmetric or not. Defaults to False.
            quant_axis (int, optional): Quantization axis. Defaults to -1.
            moving_avg (bool, optional): Whether to use moving average to update min/max values. Defaults to False.
        """
        super(AsymCacheKVObserverLayer, self).__init__(
            layer,
            quant_bits=quant_bits,
            sign=True,
            symmetric=symmetric,
            quant_axis=quant_axis)
        self.quant_bits = quant_bits
        self.qmin, self.qmax = self.qmin_qmax
        self._layer = layer
        self._max = None
        self._min = None
        self._max_global = None
        self._min_global = None

        self._scale = None
        self._zero_point = None
        self._moving_avg = moving_avg
        self._range_update_factor_min = 0.001
        self._current_iters = 0
        self._quant_axis = quant_axis

    def forward(self, inputs):
        """
            计算输入数据的绝对最大值，并返回原始数据。
        如果当前模型已经被初始化过，则会更新全局最小和最大值。
        
        Args:
            inputs (Tensor): 输入数据，形状为（N，C，H，W）或者（N，C）。
        
        Returns:
            Tensor: 返回原始输入数据。
        
        Raises:
            无。
        """
        self._min, self._max, self._min_global, self._max_global = self._cal_abs_max(inputs)
        return inputs

    def _cal_abs_max(self, inputs):
        """
            计算输入的绝对值最大值，并返回。
        如果是第一次调用，则将最大值和最小值设置为None。
        
        Args:
            inputs (Tensor): 输入张量，形状为[N, C, H, W]或者[N, H, W]。
        
        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]:
            返回四个Tensor，分别表示：
                1. min_values (Tensor): 输入张量中每个通道的最小值，形状为[N, C]或者[N]。
                2. max_values (Tensor): 输入张量中每个通道的最大值，形状为[N, C]或者[N]。
                3. min_values_global (Tensor): 输入张量中所有通道的最小值，形状为[N, C]或者[N]。
                4. max_values_global (Tensor): 输入张量中所有通道的最大值，形状为[N, C]或者[N]。
        """

        reduce_axis = tuple(
            [i for i in range(len(inputs.shape)) if i != self.quant_axis()])

        max_values = paddle.max(inputs.cast("float32"), axis=reduce_axis)
        min_values = paddle.min(inputs.cast("float32"), axis=reduce_axis)
        max_values_global = max_values
        min_values_global = min_values

        self._current_iters += 1
        if self._max is not None:
            if self._moving_avg:
                # exponential moving average update
                update_factor = 1.0 / self._current_iters
                update_factor = max(update_factor, self._range_update_factor_min)
                max_values = self._max * (1 - update_factor) + max_values * update_factor
                min_values = self._min * (1 - update_factor) + min_values * update_factor
            else:
                # normal average 
                max_values = (self._max * (self._current_iters - 1) + max_values) / self._current_iters
                min_values = (self._min * (self._current_iters - 1) + min_values) / self._current_iters

            max_values_global = paddle.maximum(self._max_global, max_values_global)
            min_values_global = paddle.minimum(self._min_global, min_values_global)

        return min_values, max_values, min_values_global, max_values_global

    def min_value(self):
        """
            返回最小值。
        
        Args:
            None
        
        Returns:
            float (float): 最小值。
        
        Raises:
            None
        """
        return self._min

    def max_value(self):
        """
            返回当前最大值。
        
        Args:
            无参数。
        
        Returns:
            float (float): 当前最大值。
        
        Raises:
            无异常抛出。
        """
        return self._max

    def cal_thresholds(self):
        """ Compute thresholds for MAX function.
        """
        self._scale, self._zero_point = self.cal_scales_zero_points()

    def scales(self):
        """ Return output scales.
        """
        if self._scale is None:
            self.cal_thresholds()
        return self._scale

    def zero_points(self):
        """ Return output zero points.
        """
        if self._zero_point is None:
            self.cal_thresholds()
        return self._zero_point