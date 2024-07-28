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

from abc import abstractmethod

import paddle
from paddle import nn


class Intervention(nn.Layer):
    """Intervention the original representations."""

    def __init__(self, **kwargs):
        super().__init__()
        self.trainable = False
        self.is_source_constant = False

        self.keep_last_dim = kwargs["keep_last_dim"] if "keep_last_dim" in kwargs else False

        if "embed_dim" in kwargs and kwargs["embed_dim"] is not None:
            self.register_buffer("embed_dim", paddle.to_tensor(kwargs["embed_dim"]))
            self.register_buffer("interchange_dim", paddle.to_tensor(kwargs["embed_dim"]))
        else:
            self.embed_dim = None
            self.interchange_dim = None

        if "source_representation" in kwargs and kwargs["source_representation"] is not None:
            self.is_source_constant = True
            self.register_buffer("source_representation", kwargs["source_representation"])
        else:
            if "hidden_source_representation" in kwargs and kwargs["hidden_source_representation"] is not None:
                self.is_source_constant = True
            else:
                self.source_representation = None

    def set_interchange_dim(self, interchange_dim):
        if isinstance(interchange_dim, int):
            self.interchange_dim = paddle.to_tensor(interchange_dim)
        else:
            self.interchange_dim = interchange_dim

    @abstractmethod
    def forward(
        self,
        base,
    ):
        pass


class SourcelessIntervention(Intervention):
    """No source."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_source_constant = True
