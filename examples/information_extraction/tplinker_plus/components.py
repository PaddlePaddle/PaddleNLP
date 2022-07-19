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
import paddle.nn.functional as F


class LayerNorm(nn.Layer):

    def __init__(self,
                 input_dim,
                 cond_dim=0,
                 center=True,
                 scale=True,
                 epsilon=None,
                 conditional=False,
                 hidden_units=None,
                 hidden_activation="linear",
                 hidden_initializer="xaiver",
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
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.bias = paddle.create_parameter(shape=[input_dim],
                                                dtype="float32")
        if self.scale:
            self.weight = paddle.create_parameter(shape=[input_dim],
                                                  dtype="float32")

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(
                    in_features=self.cond_dim,
                    out_features=self.hidden_units,
                    bias_attr=False,
                )
            if self.center:
                self.bias_dense = nn.Linear(in_features=self.cond_dim,
                                            out_features=input_dim,
                                            bias_attr=False)
            if self.scale:
                self.weight_dense = nn.Linear(in_features=self.cond_dim,
                                              out_features=input_dim,
                                              bias_attr=False)

        self.initialize_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == "normal":
                    self.hidden_dense.weight = nn.initializer.Normal()
                elif self.hidden_initializer == "xavier":  # glorot_uniform
                    self.hidden_dense.weight = nn.initializer.XavierUniform()
            # 下面这两个为什么都初始化为0呢?
            # 为了防止扰乱原来的预训练权重，两个变换矩阵可以全零初始化（单层神经网络可以用全零初始化，连续的多层神经网络才不应当用全零初始化），这样在初始状态，模型依然保持跟原来的预训练模型一致。
            if self.center:
                self.bias_dense.weight_attr = nn.initializer.Constant(0)
            if self.scale:
                self.weight_dense.weight_attr = nn.initializer.Constant(0)

    def forward(self, inputs, cond=None):
        """
        如果是条件Layer Norm，则cond不是None
        """
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            # for _ in range(K.ndim(inputs) - K.ndim(cond)): # K.ndim: 以整数形式返回张量中的轴数。
            # TODO: 这两个为什么有轴数差呢？ 为什么在 dim=1 上增加维度??
            # 为了保持维度一致，cond可以是（batch_size, cond_dim）
            for _ in range(inputs.ndim - cond.ndim):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)

            # cond在加入bias和weight之前做一次线性变换，以保证与input维度一致
            if self.center:
                bias = self.bias_dense(cond) + self.bias
            if self.scale:
                weight = self.weight_dense(cond) + self.weight
        else:
            if self.center:
                bias = self.bias
            if self.scale:
                weight = self.weight

        outputs = inputs
        if self.center:
            mean = paddle.mean(outputs, axis=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = paddle.mean(outputs**2, axis=-1).unsqueeze(-1)
            std = (variance + self.epsilon)**2
            outputs = outputs / std
            outputs = outputs * weight
        if self.center:
            outputs = outputs + bias
        return outputs


class HandshakingKernel(nn.Layer):

    def __init__(self, hidden_size, shaking_type, only_look_after=True):
        super().__init__()
        self.shaking_type = shaking_type
        self.only_look_after = only_look_after

        if "cat" in shaking_type:
            self.cat_fc = nn.Linear(hidden_size * 2, hidden_size)
        if "cln" in shaking_type:
            self.tp_cln = LayerNorm(hidden_size, hidden_size, conditional=True)
        if "lstm" in shaking_type:
            assert only_look_after is True
            self.lstm4span = nn.LSTM(hidden_size,
                                     hidden_size,
                                     num_layers=1,
                                     direction="forward")

    def upper_reg2seq(self, tensor):
        """
        drop lower region and flat upper region to sequence
        :param tensor: (batch_size, matrix_size, matrix_size, hidden_size)
        :return: (batch_size, matrix_size + ... + 1, hidden_size)
        """
        bs, matrix_size, matrix_size, hidden_size = tensor.shape
        mask = paddle.triu(paddle.ones(shape=[matrix_size, matrix_size]))
        mask = paddle.cast(mask, "bool")[None, :, :, None]
        mask = paddle.expand(mask,
                             shape=[bs, matrix_size, matrix_size, hidden_size])
        return tensor.masked_select(mask).reshape([bs, -1, hidden_size])

    def forward(self, seq_hiddens):
        """
        seq_hiddens: (batch_size, seq_len, hidden_size_x)
        return:
            if only look after:
                shaking_hiddenss: (batch_size, (1 + seq_len) * seq_len / 2, hidden_size); e.g. (32, 5+4+3+2+1, 5)
            else:
                shaking_hiddenss: (batch_size, seq_len * seq_len, hidden_size)
        """
        seq_len = seq_hiddens.shape[1]

        guide = paddle.tile(seq_hiddens[:, :, None, :],
                            repeat_times=[1, 1, seq_len, 1])
        visible = paddle.transpose(guide, [0, 2, 1, 3])

        shaking_pre = None

        def add_presentation(all_prst, prst):
            if all_prst is None:
                all_prst = prst
            else:
                all_prst += prst
            return all_prst

        if self.only_look_after:
            if "lstm" in self.shaking_type:
                batch_size, _, matrix_size, vis_hidden_size = visible.shape
                # mask lower triangle
                mask = paddle.tril(
                    paddle.ones(shape=[matrix_size, matrix_size]), -1)
                mask = paddle.cast(mask, "bool")[None, :, :, None]
                # visible4lstm: (batch_size * matrix_size, matrix_size, hidden_size)
                visible4lstm = visible.masked_fill(mask, 0).flatten(0, 1)

                span_pre = self.lstm4span(visible4lstm)[0]
                span_pre = span_pre.reshape(batch_size, matrix_size,
                                            matrix_size, vis_hidden_size)
                # drop lower triangle and convert matrix to sequence
                # span_pre: (batch_size, shaking_seq_len, hidden_size)
                span_pre = self.upper_reg2seq(span_pre)
                shaking_pre = add_presentation(shaking_pre, span_pre)

            # guide, visible: (batch_size, shaking_seq_len, hidden_size)
            guide = self.upper_reg2seq(guide)
            visible = self.upper_reg2seq(visible)

        if "cat" in self.shaking_type:
            tp_cat_pre = paddle.concat([guide, visible], axis=-1)
            tp_cat_pre = F.relu(self.cat_fc(tp_cat_pre))
            shaking_pre = add_presentation(shaking_pre, tp_cat_pre)

        if "cln" in self.shaking_type:
            tp_cln_pre = self.tp_cln(visible, guide)
            shaking_pre = add_presentation(shaking_pre, tp_cln_pre)

        return shaking_pre
