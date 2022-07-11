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


class HandshakingKernel(nn.Layer):

    def __init__(self, hidden_size, inner_enc_type):
        super().__init__()
        self.combine_fc = nn.Linear(hidden_size * 3, hidden_size)

        self.inner_enc_type = inner_enc_type
        if inner_enc_type == "mix_pooling":
            self.lamtha = paddle.create_parameter(paddle.rand(hidden_size))
        elif inner_enc_type == "lstm":
            self.inner_context_lstm = nn.LSTM(hidden_size,
                                              hidden_size,
                                              num_layers=1,
                                              direction="forward")

    def enc_inner_hiddens(self, seq_hiddens, inner_enc_type="lstm"):
        # seq_hiddens: (batch_size, seq_len, hidden_size)
        def pool(seqence, pooling_type):
            if pooling_type == "mean_pooling":
                pooling = paddle.mean(seqence, axis=-2)
            elif pooling_type == "max_pooling":
                pooling, _ = paddle.max(seqence, axis=-2)
            elif pooling_type == "mix_pooling":
                pooling = self.lamtha * paddle.mean(seqence, axis=-2) + (
                    1 - self.lamtha) * paddle.max(seqence, axis=-2)[0]
            return pooling

        if "pooling" in inner_enc_type:
            inner_context = paddle.stack([
                pool(seq_hiddens[:, :i + 1, :], inner_enc_type)
                for i in range(seq_hiddens.size()[1])
            ],
                                         axis=1)
        elif inner_enc_type == "lstm":
            inner_context, _ = self.inner_context_lstm(seq_hiddens)

        return inner_context

    def forward(self, seq_hiddens):
        '''
        seq_hiddens: (batch_size, seq_len, hidden_size)
        return:
            shaking_hiddenss: (batch_size, (1 + seq_len) * seq_len / 2, hidden_size) (32, 5+4+3+2+1, 5)
        '''
        seq_len = seq_hiddens.shape[-2]
        shaking_hiddens_list = []
        for ind in range(seq_len):
            hidden_each_step = seq_hiddens[:, ind, :]
            visible_hiddens = seq_hiddens[:, ind:, :]  # ind: only look back
            repeat_hiddens = paddle.tile(hidden_each_step[:, None, :],
                                         repeat_times=[1, seq_len - ind, 1])

            inner_context = self.enc_inner_hiddens(visible_hiddens,
                                                   self.inner_enc_type)

            shaking_hiddens = paddle.concat(
                [repeat_hiddens, visible_hiddens, inner_context], axis=-1)
            shaking_hiddens = paddle.tanh(self.combine_fc(shaking_hiddens))

            shaking_hiddens_list.append(shaking_hiddens)
        long_shaking_hiddens = paddle.concat(shaking_hiddens_list, axis=1)
        return long_shaking_hiddens
