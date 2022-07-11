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


class LayerNorm(nn.Layer):

    def __init__(self,
                 input_dim,
                 cond_dim=0,
                 center=True,
                 scale=True,
                 epsilon=None,
                 conditional=False,
                 hidden_units=None,
                 hidden_activation='linear',
                 hidden_initializer='xaiver',
                 **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        # self.hidden_activation = activations.get(hidden_activation) keras中activation为linear时，返回原tensor,unchanged
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = paddle.create_parameter(paddle.zeros(input_dim))
        if self.scale:
            self.gamma = paddle.create_parameter(paddle.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim,
                                              out_features=self.hidden_units,
                                              bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim,
                                            out_features=input_dim,
                                            bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim,
                                             out_features=input_dim,
                                             bias=False)

    def forward(self, inputs, cond=None):
        """
            如果是条件Layer Norm，则cond不是None
        """
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)

            # cond在加入beta和gamma之前做一次线性变换，以保证与input维度一致
            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = paddle.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = paddle.mean(outputs**2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon)**2
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs


class HandshakingKernel(nn.Layer):

    def __init__(self, hidden_size, shaking_type, inner_enc_type):
        super().__init__()
        self.shaking_type = shaking_type
        if shaking_type == "cat":
            self.combine_fc = nn.Linear(hidden_size * 2, hidden_size)
        elif shaking_type == "cat_plus":
            self.combine_fc = nn.Linear(hidden_size * 3, hidden_size)
        elif shaking_type == "cln":
            self.tp_cln = LayerNorm(hidden_size, hidden_size, conditional=True)
        elif shaking_type == "cln_plus":
            self.tp_cln = LayerNorm(hidden_size, hidden_size, conditional=True)
            self.inner_context_cln = LayerNorm(hidden_size,
                                               hidden_size,
                                               conditional=True)

        self.inner_enc_type = inner_enc_type
        if inner_enc_type == "mix_pooling":
            self.lamtha = paddle.create_parameter(paddle.rand(hidden_size))
        elif inner_enc_type == "lstm":
            self.inner_context_lstm = nn.LSTM(hidden_size,
                                              hidden_size,
                                              num_layers=1,
                                              bidirectional=False,
                                              batch_first=True)

    def enc_inner_hiddens(self, seq_hiddens, inner_enc_type="lstm"):
        # seq_hiddens: (batch_size, seq_len, hidden_size)
        def pool(seqence, pooling_type):
            if pooling_type == "mean_pooling":
                pooling = paddle.mean(seqence, dim=-2)
            elif pooling_type == "max_pooling":
                pooling, _ = paddle.max(seqence, dim=-2)
            elif pooling_type == "mix_pooling":
                pooling = self.lamtha * paddle.mean(seqence, dim=-2) + (
                    1 - self.lamtha) * paddle.max(seqence, dim=-2)[0]
            return pooling

        if "pooling" in inner_enc_type:
            inner_context = paddle.stack([
                pool(seq_hiddens[:, :i + 1, :], inner_enc_type)
                for i in range(seq_hiddens.size()[1])
            ],
                                         dim=1)
        elif inner_enc_type == "lstm":
            inner_context, _ = self.inner_context_lstm(seq_hiddens)

        return inner_context

    def forward(self, seq_hiddens):
        '''
        seq_hiddens: (batch_size, seq_len, hidden_size)
        return:
            shaking_hiddenss: (batch_size, (1 + seq_len) * seq_len / 2, hidden_size) (32, 5+4+3+2+1, 5)
        '''
        seq_len = seq_hiddens.size()[-2]
        shaking_hiddens_list = []
        for ind in range(seq_len):
            hidden_each_step = seq_hiddens[:, ind, :]
            visible_hiddens = seq_hiddens[:, ind:, :]  # ind: only look back
            repeat_hiddens = hidden_each_step[:, None, :].repeat(
                1, seq_len - ind, 1)

            if self.shaking_type == "cat":
                shaking_hiddens = paddle.cat([repeat_hiddens, visible_hiddens],
                                             dim=-1)
                shaking_hiddens = paddle.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cat_plus":
                inner_context = self.enc_inner_hiddens(visible_hiddens,
                                                       self.inner_enc_type)
                shaking_hiddens = paddle.cat(
                    [repeat_hiddens, visible_hiddens, inner_context], dim=-1)
                shaking_hiddens = paddle.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cln":
                shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)
            elif self.shaking_type == "cln_plus":
                inner_context = self.enc_inner_hiddens(visible_hiddens,
                                                       self.inner_enc_type)
                shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)
                shaking_hiddens = self.inner_context_cln(
                    shaking_hiddens, inner_context)

            shaking_hiddens_list.append(shaking_hiddens)
        long_shaking_hiddens = paddle.cat(shaking_hiddens_list, dim=1)
        return long_shaking_hiddens
