# -*- coding: utf-8 -*-
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid.layers as layers
import paddle.fluid as fluid
import numpy as np
from paddle.fluid import ParamAttr
from paddle.fluid.layers import RNNCell, LSTMCell, rnn, BeamSearchDecoder, dynamic_decode

INF = 1. * 1e5
alpha = 0.6
uniform_initializer = lambda x: fluid.initializer.UniformInitializer(low=-x, high=x)
zero_constant = fluid.initializer.Constant(0.0)


class EncoderCell(RNNCell):
    def __init__(
            self,
            num_layers,
            hidden_size,
            dropout_prob=0.,
            init_scale=0.1, ):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.lstm_cells = []

        param_attr = ParamAttr(initializer=uniform_initializer(init_scale))
        bias_attr = ParamAttr(initializer=zero_constant)
        for i in range(num_layers):
            self.lstm_cells.append(LSTMCell(hidden_size, param_attr, bias_attr))

    def call(self, step_input, states):
        new_states = []
        for i in range(self.num_layers):
            out, new_state = self.lstm_cells[i](step_input, states[i])
            step_input = layers.dropout(
                out,
                self.dropout_prob,
                dropout_implementation="upscale_in_train"
            ) if self.dropout_prob > 0. else out
            new_states.append(new_state)
        return step_input, new_states

    @property
    def state_shape(self):
        return [cell.state_shape for cell in self.lstm_cells]


class DecoderCell(RNNCell):
    def __init__(self, num_layers, hidden_size, dropout_prob=0.,
                 init_scale=0.1):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.lstm_cells = []
        self.init_scale = init_scale
        param_attr = ParamAttr(initializer=uniform_initializer(init_scale))
        bias_attr = ParamAttr(initializer=zero_constant)
        for i in range(num_layers):
            self.lstm_cells.append(LSTMCell(hidden_size, param_attr, bias_attr))

    def call(self, step_input, states):
        new_lstm_states = []
        for i in range(self.num_layers):
            out, new_lstm_state = self.lstm_cells[i](step_input, states[i])
            step_input = layers.dropout(
                out,
                self.dropout_prob,
                dropout_implementation="upscale_in_train"
            ) if self.dropout_prob > 0. else out
            new_lstm_states.append(new_lstm_state)
        return step_input, new_lstm_states


class BaseModel(object):
    def __init__(self,
                 hidden_size,
                 src_vocab_size,
                 tar_vocab_size,
                 batch_size,
                 num_layers=1,
                 init_scale=0.1,
                 dropout=None,
                 beam_start_token=1,
                 beam_end_token=2,
                 beam_max_step_num=100):

        self.hidden_size = hidden_size
        self.src_vocab_size = src_vocab_size
        self.tar_vocab_size = tar_vocab_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.init_scale = init_scale
        self.dropout = dropout
        self.beam_start_token = beam_start_token
        self.beam_end_token = beam_end_token
        self.beam_max_step_num = beam_max_step_num
        self.src_embeder = lambda x: fluid.embedding(
            input=x,
            size=[self.src_vocab_size, self.hidden_size],
            dtype='float32',
            is_sparse=False,
            param_attr=fluid.ParamAttr(
                name='source_embedding',
                initializer=uniform_initializer(init_scale)))

        self.tar_embeder = lambda x: fluid.embedding(
            input=x,
            size=[self.tar_vocab_size, self.hidden_size],
            dtype='float32',
            is_sparse=False,
            param_attr=fluid.ParamAttr(
                name='target_embedding',
                initializer=uniform_initializer(init_scale)))

    def _build_data(self):
        self.src = fluid.data(name="src", shape=[None, None], dtype='int64')
        self.src_sequence_length = fluid.data(
            name="src_sequence_length", shape=[None], dtype='int32')

        self.tar = fluid.data(name="tar", shape=[None, None], dtype='int64')
        self.tar_sequence_length = fluid.data(
            name="tar_sequence_length", shape=[None], dtype='int32')
        self.label = fluid.data(
            name="label", shape=[None, None, 1], dtype='int64')

    def _embedding(self):
        self.src_emb = self.src_embeder(self.src)
        self.tar_emb = self.tar_embeder(self.tar)

    def _build_encoder(self):
        enc_cell = EncoderCell(self.num_layers, self.hidden_size, self.dropout,
                               self.init_scale)
        self.enc_output, enc_final_state = rnn(
            cell=enc_cell,
            inputs=self.src_emb,
            sequence_length=self.src_sequence_length)
        return self.enc_output, enc_final_state

    def _build_decoder(self, enc_final_state, mode='train', beam_size=10):

        dec_cell = DecoderCell(self.num_layers, self.hidden_size, self.dropout,
                               self.init_scale)
        output_layer = lambda x: layers.fc(x,
                                            size=self.tar_vocab_size,
                                            num_flatten_dims=len(x.shape) - 1,
                                            param_attr=fluid.ParamAttr(name="output_w",
                                                initializer=uniform_initializer(self.init_scale)),
                                            bias_attr=False)

        if mode == 'train':
            dec_output, dec_final_state = rnn(cell=dec_cell,
                                              inputs=self.tar_emb,
                                              initial_states=enc_final_state)

            dec_output = output_layer(dec_output)

            return dec_output
        elif mode == 'beam_search':
            beam_search_decoder = BeamSearchDecoder(
                dec_cell,
                self.beam_start_token,
                self.beam_end_token,
                beam_size,
                embedding_fn=self.tar_embeder,
                output_fn=output_layer)

            outputs, _ = dynamic_decode(
                beam_search_decoder,
                inits=enc_final_state,
                max_step_num=self.beam_max_step_num)
            return outputs

    def _compute_loss(self, dec_output):
        loss = layers.softmax_with_cross_entropy(
            logits=dec_output, label=self.label, soft_label=False)
        loss = layers.unsqueeze(loss, axes=[2])

        max_tar_seq_len = layers.shape(self.tar)[1]
        tar_mask = layers.sequence_mask(
            self.tar_sequence_length, maxlen=max_tar_seq_len, dtype='float32')
        loss = layers.elementwise_mul(loss, tar_mask, axis=0)
        loss = layers.reduce_mean(loss, dim=[0])
        loss = layers.reduce_sum(loss)
        return loss

    def _beam_search(self, enc_last_hidden, enc_last_cell):
        pass

    def build_graph(self, mode='train', beam_size=10):
        if mode == 'train' or mode == 'eval':
            self._build_data()
            self._embedding()
            enc_output, enc_final_state = self._build_encoder()
            dec_output = self._build_decoder(enc_final_state)

            loss = self._compute_loss(dec_output)
            return loss
        elif mode == "beam_search" or mode == 'greedy_search':
            self._build_data()
            self._embedding()
            enc_output, enc_final_state = self._build_encoder()
            dec_output = self._build_decoder(
                enc_final_state, mode=mode, beam_size=beam_size)

            return dec_output
        else:
            print("not support mode ", mode)
            raise Exception("not support mode: " + mode)
