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

import math
import numpy as np
import paddle
import paddle.nn as nn

from components import HandshakingKernel


class TPLinkerPlus(nn.Layer):
    "Network for TPLinkerPlus"

    def __init__(self,
                 encoder,
                 num_tags,
                 shaking_type="cln",
                 tok_pair_sample_rate=1):
        super().__init__()
        self.encoder = encoder
        self.shaking_type = shaking_type

        shaking_hidden_size = encoder.config["hidden_size"]

        # handshaking kernel
        self.handshaking_kernel = HandshakingKernel(shaking_hidden_size,
                                                    shaking_type)
        self.out_proj = nn.Linear(shaking_hidden_size, num_tags)

    def forward(self, input_ids, attention_mask):
        # input_ids, attention_mask, token_type_ids: (batch_size, seq_len)
        context_outputs = self.encoder(input_ids, attention_mask=attention_mask)
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]

        # shaking_hiddens: (batch_size, shaking_seq_len, hidden_size)
        shaking_hiddens = self.handshaking_kernel(last_hidden_state)

        # shaking_logits: (batch_size, shaking_seq_len, tag_size)
        shaking_logits = self.out_proj(shaking_hiddens)

        return shaking_logits
