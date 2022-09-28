# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from model.dropouts import SharedDropout
from model.encoder import LSTMEncoder, LSTMByWPEncoder, ErnieEncoder


class BiAffineParser(nn.Layer):
    """DDParser"""

    def __init__(self,
                 encoding_model,
                 feat,
                 n_rels,
                 n_feats,
                 n_words,
                 pad_index,
                 eos_index,
                 pretrained_model=None,
                 n_mlp_arc=500,
                 n_mlp_rel=100,
                 mlp_dropout=0.33):
        super(BiAffineParser, self).__init__()
        self.pad_index = pad_index
        self.eos_index = eos_index

        if encoding_model == "lstm":
            self.embed = LSTMEncoder(feat, n_feats, n_words)
        elif encoding_model == "lstm-pe":
            self.embed = LSTMByWPEncoder(n_words, pad_index)
        else:
            self.embed = ErnieEncoder(pad_index, pretrained_model)

        # MLP layer
        self.mlp_arc_h = MLP(n_in=self.embed.mlp_input_size,
                             n_out=n_mlp_arc,
                             dropout=mlp_dropout)
        self.mlp_arc_d = MLP(n_in=self.embed.mlp_input_size,
                             n_out=n_mlp_arc,
                             dropout=mlp_dropout)
        self.mlp_rel_h = MLP(n_in=self.embed.mlp_input_size,
                             n_out=n_mlp_rel,
                             dropout=mlp_dropout)
        self.mlp_rel_d = MLP(n_in=self.embed.mlp_input_size,
                             n_out=n_mlp_rel,
                             dropout=mlp_dropout)

        # Biaffine layer
        self.arc_attn = BiAffine(n_in=n_mlp_arc, bias_x=True, bias_y=False)
        self.rel_attn = BiAffine(n_in=n_mlp_rel,
                                 n_out=n_rels,
                                 bias_x=True,
                                 bias_y=True)

    def forward(self, words, feats):

        words, x = self.embed(words, feats)
        mask = paddle.logical_and(words != self.pad_index,
                                  words != self.eos_index)

        arc_h = self.mlp_arc_h(x)
        arc_d = self.mlp_arc_d(x)
        rel_h = self.mlp_rel_h(x)
        rel_d = self.mlp_rel_d(x)

        # Get arc and rel scores from the bilinear attention
        # Shape: (batch_size, seq_len, seq_len)
        s_arc = self.arc_attn(arc_d, arc_h)
        # Shape: (batch_size, seq_len, seq_len, n_rels)
        s_rel = paddle.transpose(self.rel_attn(rel_d, rel_h), perm=[0, 2, 3, 1])
        # Set the scores that exceed the length of each sentence to -1e5
        s_arc_mask = paddle.unsqueeze(mask, 1)
        s_arc = s_arc * s_arc_mask + paddle.scale(paddle.cast(
            s_arc_mask, 'int32'),
                                                  scale=1e5,
                                                  bias=-1,
                                                  bias_after_scale=False)
        return s_arc, s_rel, words


class MLP(nn.Layer):
    """MLP"""

    def __init__(self, n_in, n_out, dropout=0):
        super(MLP, self).__init__()

        self.linear = nn.Linear(
            n_in,
            n_out,
            weight_attr=nn.initializer.XavierNormal(),
        )
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = SharedDropout(p=dropout)

    def forward(self, x):
        # Shape: (batch_size, output_size)
        x = self.linear(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        return x


class BiAffine(nn.Layer):
    """BiAffine"""

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(BiAffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = self.create_parameter(
            shape=[n_out, n_in + bias_x, n_in + bias_y], dtype="float32")

    def forward(self, x, y):
        if self.bias_x:
            x = paddle.concat([x, paddle.ones_like(x[:, :, :1])], axis=-1)
        if self.bias_y:
            y = paddle.concat([y, paddle.ones_like(x[:, :, :1])], axis=-1)
        # Shape x: (batch_size, num_tokens, input_size + bias_x)
        b = x.shape[0]
        o = self.weight.shape[0]
        # Shape x: (batch_size, output_size, num_tokens, input_size + bias_x)
        x = paddle.expand(paddle.unsqueeze(x, axis=1),
                          shape=(x.shape[0], o, x.shape[1], x.shape[2]))
        # Shape y: (batch_size, output_size, num_tokens, input_size + bias_y)
        y = paddle.expand(paddle.unsqueeze(y, axis=1),
                          shape=(y.shape[0], o, y.shape[1], y.shape[2]))
        # Shape weight: (batch_size, output_size, input_size + bias_x, input_size + bias_y)
        weight = paddle.expand(paddle.unsqueeze(self.weight, axis=0),
                               shape=(b, self.weight.shape[0],
                                      self.weight.shape[1],
                                      self.weight.shape[2]))

        # Shape: (batch_size, output_size, num_tokens, num_tokens)
        s = paddle.matmul(paddle.matmul(x, weight),
                          paddle.transpose(y, perm=[0, 1, 3, 2]))
        # Remove dim 1 if n_out == 1
        if s.shape[1] == 1:
            s = paddle.squeeze(s, axis=1)
        return s
