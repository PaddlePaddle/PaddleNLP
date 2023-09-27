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
import math

import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.nn.quant.format import ConvertibleQuantedLayer

from ...utils.quantization import QuantizationLinear


class QuantedLoRALinear(ConvertibleQuantedLayer):
    """
    The computational logic of QuantizedLoRALinear is the same as LoRALinear.
    The only difference is that its inputs are all fake quantized.

    Note:
        In order for proper quantization of this layer, we do (W + AB)x instead of Wx + ABx as in LoRALinear.
        The quanted logic is quant(W + AB)x
    """

    def __init__(self, layer: nn.Layer, q_config):
        super().__init__()
        if isinstance(layer.lora_dropout, nn.Dropout):
            raise ValueError("lora_dropout is not supported for QuantedLoRALinear")

        self.weight = layer.weight
        self.lora_A = layer.lora_A
        self.lora_B = layer.lora_B
        self.scaling = layer.scaling
        self.bias = layer.bias
        self.name = layer.name

        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = layer.merge_weights

        # For FakeQuant

        self.weight_quanter = None
        self.activation_quanter = None
        if q_config.weight is not None:
            self.weight_quanter = q_config.weight._instance(layer)
        if q_config.activation is not None:
            self.activation_quanter = q_config.activation._instance(layer)

    def forward(self, input):

        if self.merge_weights and self.merged:
            weight = self.weight
        else:
            weight = self.weight + self.lora_A @ self.lora_B * self.scaling

        quant_input = self.activation_quanter(input) if self.activation_quanter is not None else input
        quant_weight = self.weight_quanter(weight) if self.weight_quanter is not None else weight

        return self._linear_forward(quant_input, quant_weight)

    def _linear_forward(self, input, weight):
        weight = paddle.cast(weight, input.dtype)
        out = F.linear(x=input, weight=weight, bias=self.bias, name=self.name)
        return out

    def train(self):
        super().train()
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            new_weight = self.weight - self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = False

    def eval(self):
        super().eval()
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            new_weight = self.weight + self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = True

    def weights_to_quanters(self):
        return [("weight", "weight_quanter")]

    def activation_quanters(self):
        return ["activation_quanter"]


class QuantizationLoRALinear(QuantizationLinear):
    def __init__(
        self,
        in_features,
        out_features,
        quant_algo,
        dtype,
        weight_attr=None,
        scale_attr=None,
        bias_attr=None,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
    ):
        QuantizationLinear.__init__(
            self, in_features, out_features, quant_algo, dtype, weight_attr, scale_attr, bias_attr
        )
        if not isinstance(r, int) or r <= 0:
            raise ValueError("Lora rank r should be a positive integer")
        if self.quant_algo == "llm.int8":
            raise NotImplementedError("llm.int8 not yet support lora strategy.")
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        # Actual trainable parameters
        self.lora_A = self.create_parameter(
            shape=[in_features, r],
            dtype=self._dtype,
            is_bias=False,
            default_initializer=nn.initializer.KaimingUniform(negative_slope=math.sqrt(5), nonlinearity="leaky_relu"),
        )
        self.lora_B = self.create_parameter(
            shape=[r, out_features],
            dtype=self._dtype,
            is_bias=False,
            default_initializer=nn.initializer.Constant(value=0.0),
        )
        self.scaling = self.lora_alpha / self.r

    def forward(self, x: paddle.Tensor):
        result = super().forward(x)
        result += (self.lora_dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
        return result
