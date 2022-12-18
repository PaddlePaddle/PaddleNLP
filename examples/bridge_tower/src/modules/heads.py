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

import paddle
import paddle.nn as nn

from .bert_model import BertPredictionHeadTransform


class LinkTower(nn.Layer):
    def __init__(self, config):
        super(LinkTower, self).__init__()
        self.LayerNorm = nn.LayerNorm(config["hidden_size"])

    def forward(self, hidden_states, cross_modal_hidden_states):
        return self.LayerNorm(hidden_states + cross_modal_hidden_states)


class Pooler(nn.Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ITCHead(nn.Layer):
    def __init__(self, hidden_size, embed_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, embed_size)

    def forward(self, x):
        return self.fc(x)


class ITMHead(nn.Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        return self.fc(x)


class MLMHead(nn.Layer):
    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias_attr=False)
        self.bias = paddle.create_parameter(
            shape=[config.vocab_size], dtype="float32", default_initializer=nn.initializer.Constant(value=0.0)
        )
        if weight is not None:
            # self.decoder.weight = weight
            self.decoder.weight.set_value(weight)

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x
