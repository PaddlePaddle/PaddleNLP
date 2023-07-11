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

import paddle
import paddle.nn as nn

from paddlenlp.transformers.t5.configuration import T5Config
from paddlenlp.transformers.t5.modeling import T5Block, T5LayerNorm

from ...configuration_utils import ConfigMixin, ModuleUtilsMixin, register_to_config
from ...models import ModelMixin


class SpectrogramContEncoder(ModelMixin, ConfigMixin, ModuleUtilsMixin):
    @register_to_config
    def __init__(
        self,
        input_dims: int,
        targets_context_length: int,
        d_model: int,
        dropout_rate: float,
        num_layers: int,
        num_heads: int,
        d_kv: int,
        d_ff: int,
        feed_forward_proj: str,
        is_decoder: bool = False,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_features=input_dims, out_features=d_model, bias_attr=False)
        self.position_encoding = nn.Embedding(targets_context_length, d_model)
        self.position_encoding.weight.stop_gradient = True
        self.dropout_pre = nn.Dropout(p=dropout_rate)
        t5config = T5Config(
            d_model=d_model,
            num_heads=num_heads,
            d_kv=d_kv,
            d_ff=d_ff,
            feed_forward_proj=feed_forward_proj,
            dropout_rate=dropout_rate,
            is_decoder=is_decoder,
            is_encoder_decoder=False,
        )
        self.encoders = nn.LayerList()
        for lyr_num in range(num_layers):
            lyr = T5Block(t5config)
            self.encoders.append(lyr)
        self.layer_norm = T5LayerNorm(d_model)
        self.dropout_post = nn.Dropout(p=dropout_rate)

    def forward(self, encoder_inputs, encoder_inputs_mask):

        # terminal relative positional encodings
        x = self.input_proj(encoder_inputs)
        max_positions = encoder_inputs.shape[1]
        input_positions = paddle.arange(end=max_positions)

        seq_lens = encoder_inputs_mask.sum(axis=-1)
        input_positions = paddle.roll(x=input_positions.unsqueeze(axis=0), shifts=tuple(seq_lens.tolist()), axis=0)
        x += self.position_encoding(input_positions)
        x = self.dropout_pre(x)

        # inverted the attention mask
        input_shape = encoder_inputs.shape
        extended_attention_mask = self.get_extended_attention_mask(encoder_inputs_mask, input_shape)

        for lyr in self.encoders:
            x = lyr(x, extended_attention_mask)[0]
        x = self.layer_norm(x)
        return self.dropout_post(x), encoder_inputs_mask
