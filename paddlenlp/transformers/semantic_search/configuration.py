# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
""" GPT model configuration"""
from __future__ import annotations

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = []


GPT_PRETRAINED_INIT_CONFIGURATION = {}

GPT_PRETRAINED_RESOURCE_FILES_MAP = {}


class EncoderConfig(PretrainedConfig):
    model_type = "encoder"

    pretrained_init_configuration = GPT_PRETRAINED_INIT_CONFIGURATION

    def __init__(self, model_name: str, **kwargs):
        super().__init__(**kwargs)

        self.model_name = model_name
