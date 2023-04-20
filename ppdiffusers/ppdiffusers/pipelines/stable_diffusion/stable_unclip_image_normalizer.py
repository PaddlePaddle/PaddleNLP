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
import paddle.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...models.modeling_utils import ModelMixin


class StableUnCLIPImageNormalizer(ModelMixin, ConfigMixin):
    """
    This class is used to hold the mean and standard deviation of the CLIP embedder used in stable unCLIP.

    It is used to normalize the image embeddings before the noise is applied and un-normalize the noised image
    embeddings.
    """

    @register_to_config
    def __init__(
        self,
        embedding_dim: int = 768,
    ):
        super().__init__()

        self.mean = self.create_parameter(
            (1, embedding_dim), dtype=paddle.get_default_dtype(), default_initializer=nn.initializer.Constant(0.0)
        )
        self.std = self.create_parameter(
            (1, embedding_dim), dtype=paddle.get_default_dtype(), default_initializer=nn.initializer.Constant(1.0)
        )

    def scale(self, embeds):
        embeds = (embeds - self.mean) * 1.0 / self.std
        return embeds

    def unscale(self, embeds):
        embeds = (embeds * self.std) + self.mean
        return embeds
