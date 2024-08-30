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

from typing import Dict

import paddle
from experimental.layers.cache_kv import CacheKVMatMul
from paddleslim.quant.observers.uniform import UniformObserver

CHANNEL_AXIS: Dict[type, int] = {
    paddle.nn.Conv2D: 0,
    paddle.nn.Linear: 1,
    paddle.distributed.fleet.meta_parallel.ColumnParallelLinear: 1,
    paddle.distributed.fleet.meta_parallel.RowParallelLinear: 1,
    CacheKVMatMul: 1,
}


class ChannelWiseObserver(UniformObserver):
    def __init__(self, layer, quant_bits=8, sign=True, symmetric=True, quant_axis=None):
        super(ChannelWiseObserver, self).__init__(
            quant_bits=quant_bits,
            sign=sign,
            symmetric=symmetric,
        )
        if quant_axis is not None:
            self._channel_axis = quant_axis
        else:
            assert type(layer) in CHANNEL_AXIS, "Unsupported layer type: {}".format(type(layer))
            self._channel_axis = CHANNEL_AXIS[type(layer)]
        self._quant_bits = quant_bits

    def quant_axis(self):
        """Return quantization axis."""
        return self._channel_axis

    def bit_length(self):
        """Return the bit length of quantized data."""
        return self._quant_bits
