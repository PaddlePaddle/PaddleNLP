# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2022 Alibaba PAI team. All Rights Reserved.
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
import paddle.nn.functional as F

from ..gpt.modeling import GPTLMHead, GPTLMHeadModel, GPTModel
from .configuration import (
    ARTIST_PRETRAINED_INIT_CONFIGURATION,
    ARTIST_PRETRAINED_RESOURCE_FILES_MAP,
    ArtistConfig,
)

__all__ = [
    "ArtistModel",
    "ArtistForConditionalGeneration",
]

# set gelu_new
F.gelu_python = F.gelu


class ArtistModel(GPTModel):
    config_class = ArtistConfig
    pretrained_init_configuration = ARTIST_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = ARTIST_PRETRAINED_RESOURCE_FILES_MAP


class ArtistForConditionalGeneration(GPTLMHeadModel):
    """
    The ArtistT(GPT) Model with a `language modeling` head on top.

    Args:
        gpt (:class:`ArtistModel`):
            An instance of :class:`ArtistModel`.

    """

    config_class = ArtistConfig
    pretrained_init_configuration = ARTIST_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = ARTIST_PRETRAINED_RESOURCE_FILES_MAP

    def __init__(self, config: ArtistConfig):
        super().__init__(config)
        self.lm_head = GPTLMHead(config)

    @staticmethod
    def prepare_attention_mask_for_generation(input_ids, pad_token_id, eos_token_id):
        # we don't use attention_mask
        attention_mask = paddle.zeros_like(input_ids, dtype=paddle.get_default_dtype())
        return paddle.unsqueeze(attention_mask, axis=[1, 2])
