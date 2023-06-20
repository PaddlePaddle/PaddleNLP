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


from paddle.nn import Layer
from paddle.nn import functional as F
from paddle.nn.quant.format import ConvertibleQuantedLayer


class QuantedLoRALinear(ConvertibleQuantedLayer):
    """
    The computational logic of QuantizedLoRALinear is the same as LoRALinear.
    The only difference is that its inputs are all fake quantized.

    Note:
        In order for proper quantization of this layer, we do (W + AB)x instead of Wx + ABx as in LoRALinear.
        The quanted logic is quant(W + AB)x
    """

    def __init__(self, layer: Layer, q_config):
        super().__init__()
        # For Linear
        self.weight = layer.weight
        self.lora_A = layer.lora_A
        self.lora_B = layer.lora_B
        self.lora_dropout = layer.lora_dropout
        self.bias = layer.bias
        self.name = layer.name
        # For FakeQuant

        self.weight_quanter = None
        self.activation_quanter = None
        if q_config.weight is not None:
            self.weight_quanter = q_config.weight._instance(layer)
        if q_config.activation is not None:
            self.activation_quanter = q_config.activation._instance(layer)

    def forward(self, input):
        quant_input = input
        quant_weight = self.weight + self.lora_A @ self.lora_B
        if self.activation_quanter is not None:
            quant_input = self.activation_quanter(input)
        if self.weight_quanter is not None:
            quant_weight = self.weight_quanter(self.weight)
        return self._linear_forward(quant_input, quant_weight)

    def _linear_forward(self, input, weight):
        out = F.linear(x=self.lora_dropout(input), weight=weight, bias=self.bias, name=self.name)
        return out

    def weights_to_quanters(self):
        return [("weight", "weight_quanter")]

    def activation_quanters(self):
        return ["activation_quanter"]
