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
"""
Custome Attention Layer for quantization.
"""
import paddle.tensor as tensor
from paddle.nn import Layer
from paddle.nn.quant.format import ConvertibleQuantedLayer


class QuantizedCustomAttentionLayer(ConvertibleQuantedLayer):
    """
    Quantized Custom Attention Layer.
    """

    def __init__(self, layer: Layer, q_config=None):
        """
        Initialize the QuantizeWrapper class.

        Args:
            layer (Layer): The layer to be quantized.
            q_config (QuantConfig, optional): The quantization configuration. Defaults to None.
        """
        super().__init__()
        # hard code: get activation quanter from weight
        self.activation_quanter_k = q_config.weight._instance(layer)
        self.activation_quanter_v = q_config.activation._instance(layer)
        self.layer = layer
        self.quant_info = None
        layer_name = self.layer.full_name()
        self.layer_id = int(layer_name.split("_")[-1])
        self.kv_losses = {}

    def forward(self, q, config, k, v, attention_mask, output_attentions, **kwargs):
        """forward"""
        perm = [0, 2, 1, 3]  # [1, 2, 0, 3] if self.sequence_parallel else [0, 2, 1, 3]
        tmp_k = tensor.transpose(x=k, perm=perm)
        tmp_v = tensor.transpose(x=v, perm=perm)
        if self.activation_quanter_k is not None:
            tmp_k = self.activation_quanter_k(tmp_k)
        if self.activation_quanter_v is not None:
            tmp_v = self.activation_quanter_v(tmp_v)
        k = tensor.transpose(x=tmp_k, perm=perm)
        v = tensor.transpose(x=tmp_v, perm=perm)
        return self.layer(
            q,
            config,
            k,
            v,
            attention_mask,
            output_attentions,
            **kwargs,
        )

    def weights_to_quanters(self):
        """weights to quanters"""
        return []

    def activation_quanters(self):
        """activation to quanters"""
        return ["activation_quanter_k", "activation_quanter_v"]
