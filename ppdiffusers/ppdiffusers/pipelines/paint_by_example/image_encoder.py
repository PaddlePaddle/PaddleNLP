# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from paddle import nn

from paddlenlp.transformers import (
    CLIPPretrainedModel,
    CLIPVisionConfig,
    CLIPVisionModel,
)

from ...models.attention import BasicTransformerBlock
from ...utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class PaintByExampleImageEncoder(CLIPPretrainedModel):
    config_class = CLIPVisionConfig

    def __init__(self, config: CLIPVisionConfig, proj_size=None):
        super().__init__(config)
        if proj_size is not None:
            self.projection_dim = proj_size
        else:
            self.projection_dim = config.projection_dim

        self.model = CLIPVisionModel(config)
        self.mapper = PaintByExampleMapper(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)
        self.proj_out = nn.Linear(config.hidden_size, self.projection_dim)

        # uncondition for scaling
        self.uncond_vector = self.create_parameter(
            [1, 1, self.projection_dim],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Assign(paddle.rand((1, 1, self.projection_dim))),
        )

    def forward(self, pixel_values, return_uncond_vector=False):
        clip_output = self.model(pixel_values=pixel_values)
        latent_states = clip_output.pooler_output
        latent_states = self.mapper(latent_states[:, None])
        latent_states = self.final_layer_norm(latent_states)
        latent_states = self.proj_out(latent_states)
        if return_uncond_vector:
            return latent_states, self.uncond_vector

        return latent_states


class PaintByExampleMapper(nn.Layer):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        num_layers = (config.num_hidden_layers + 1) // 5
        hid_size = config.hidden_size
        num_heads = 1
        self.blocks = nn.LayerList(
            [
                BasicTransformerBlock(hid_size, num_heads, hid_size, activation_fn="gelu", attention_bias=True)
                for _ in range(num_layers)
            ]
        )

    def forward(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)

        return hidden_states
