# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn.initializer as I


class RnnLm(nn.Layer):
    def __init__(self, vocab_size, hidden_size, batch_size, num_layers=1, init_scale=0.1, dropout=0.0):
        super(RnnLm, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.init_scale = init_scale
        self.batch_size = batch_size
        self.reset_states()

        self.embedder = nn.Embedding(
            vocab_size,
            hidden_size,
            weight_attr=paddle.ParamAttr(initializer=I.Uniform(low=-init_scale, high=init_scale)),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            weight_ih_attr=paddle.ParamAttr(initializer=I.Uniform(low=-init_scale, high=init_scale)),
            weight_hh_attr=paddle.ParamAttr(initializer=I.Uniform(low=-init_scale, high=init_scale)),
        )

        self.fc = nn.Linear(
            hidden_size,
            vocab_size,
            weight_attr=paddle.ParamAttr(initializer=I.Uniform(low=-init_scale, high=init_scale)),
            bias_attr=paddle.ParamAttr(initializer=I.Uniform(low=-init_scale, high=init_scale)),
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs):
        x = inputs
        x_emb = self.embedder(x)
        x_emb = self.dropout(x_emb)

        y, (self.hidden, self.cell) = self.lstm(x_emb, (self.hidden, self.cell))
        (self.hidden, self.cell) = tuple([item.detach() for item in (self.hidden, self.cell)])
        y = self.dropout(y)
        y = self.fc(y)
        return y

    def reset_states(self):
        self.hidden = paddle.zeros(shape=[self.num_layers, self.batch_size, self.hidden_size], dtype="float32")
        self.cell = paddle.zeros(shape=[self.num_layers, self.batch_size, self.hidden_size], dtype="float32")


class CrossEntropyLossForLm(nn.Layer):
    def __init__(self):
        super(CrossEntropyLossForLm, self).__init__()

    def forward(self, y, label):
        label = paddle.unsqueeze(label, axis=2)
        loss = paddle.nn.functional.cross_entropy(input=y, label=label, reduction="none")
        loss = paddle.squeeze(loss, axis=[2])
        loss = paddle.mean(loss, axis=[0])
        loss = paddle.sum(loss)
        return loss


class UpdateModel(paddle.callbacks.Callback):
    # This callback reset model hidden states and update learning rate before each epoch begins
    def on_epoch_begin(self, epoch=None, logs=None):
        self.model.network.reset_states()
