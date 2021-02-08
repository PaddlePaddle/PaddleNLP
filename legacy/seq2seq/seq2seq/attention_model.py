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
from paddle.fluid.contrib.layers import basic_lstm, BasicLSTMUnit
from base_model import BaseModel, DecoderCell
from paddle.fluid.layers import RNNCell, LSTMCell, rnn, BeamSearchDecoder, dynamic_decode

INF = 1. * 1e5
alpha = 0.6


class AttentionDecoderCell(DecoderCell):
    def __init__(self, num_layers, hidden_size, dropout_prob=0.,
                 init_scale=0.1):
        super(AttentionDecoderCell, self).__init__(num_layers, hidden_size,
                                                   dropout_prob, init_scale)

    def attention(self, query, enc_output, mask=None):
        query = layers.unsqueeze(query, [1])
        memory = layers.fc(enc_output,
                           self.hidden_size,
                           num_flatten_dims=2,
                           param_attr=ParamAttr(
                               initializer=fluid.initializer.UniformInitializer(
                                   low=-self.init_scale, high=self.init_scale)),
                           bias_attr=False)
        attn = layers.matmul(query, memory, transpose_y=True)

        if mask:
            attn = layers.transpose(attn, [1, 0, 2])
            attn = layers.elementwise_add(attn, mask * 1000000000, -1)
            attn = layers.transpose(attn, [1, 0, 2])
        weight = layers.softmax(attn)
        weight_memory = layers.matmul(weight, memory)

        return weight_memory

    def call(self, step_input, states, enc_output, enc_padding_mask=None):
        lstm_states, input_feed = states
        new_lstm_states = []
        step_input = layers.concat([step_input, input_feed], 1)
        for i in range(self.num_layers):
            out, new_lstm_state = self.lstm_cells[i](step_input, lstm_states[i])
            step_input = layers.dropout(
                out,
                self.dropout_prob,
                dropout_implementation='upscale_in_train'
            ) if self.dropout_prob > 0 else out
            new_lstm_states.append(new_lstm_state)
        dec_att = self.attention(step_input, enc_output, enc_padding_mask)
        dec_att = layers.squeeze(dec_att, [1])
        concat_att_out = layers.concat([dec_att, step_input], 1)
        out = layers.fc(concat_att_out,
                        self.hidden_size,
                        param_attr=ParamAttr(
                            initializer=fluid.initializer.UniformInitializer(
                                low=-self.init_scale, high=self.init_scale)),
                        bias_attr=False)
        return out, [new_lstm_states, out]


class AttentionModel(BaseModel):
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
        super(AttentionModel, self).__init__(
            hidden_size,
            src_vocab_size,
            tar_vocab_size,
            batch_size,
            num_layers=num_layers,
            init_scale=init_scale,
            dropout=dropout)

    def _build_decoder(self, enc_final_state, mode='train', beam_size=10):
        output_layer = lambda x: layers.fc(x,
                                            size=self.tar_vocab_size,
                                            num_flatten_dims=len(x.shape) - 1,
                                            param_attr=fluid.ParamAttr(name="output_w",
                                                initializer=fluid.initializer.UniformInitializer(low=-self.init_scale, high=self.init_scale)),
                                            bias_attr=False)

        dec_cell = AttentionDecoderCell(self.num_layers, self.hidden_size,
                                        self.dropout, self.init_scale)
        dec_initial_states = [
            enc_final_state, dec_cell.get_initial_states(
                batch_ref=self.enc_output, shape=[self.hidden_size])
        ]
        max_src_seq_len = layers.shape(self.src)[1]
        src_mask = layers.sequence_mask(
            self.src_sequence_length, maxlen=max_src_seq_len, dtype='float32')
        enc_padding_mask = (src_mask - 1.0)
        if mode == 'train':
            dec_output, _ = rnn(cell=dec_cell,
                                inputs=self.tar_emb,
                                initial_states=dec_initial_states,
                                sequence_length=None,
                                enc_output=self.enc_output,
                                enc_padding_mask=enc_padding_mask)

            dec_output = output_layer(dec_output)

        elif mode == 'beam_search':
            output_layer = lambda x: layers.fc(x,
                                            size=self.tar_vocab_size,
                                            num_flatten_dims=len(x.shape) - 1,
                                            param_attr=fluid.ParamAttr(name="output_w"),
                                            bias_attr=False)
            beam_search_decoder = BeamSearchDecoder(
                dec_cell,
                self.beam_start_token,
                self.beam_end_token,
                beam_size,
                embedding_fn=self.tar_embeder,
                output_fn=output_layer)
            enc_output = beam_search_decoder.tile_beam_merge_with_batch(
                self.enc_output, beam_size)
            enc_padding_mask = beam_search_decoder.tile_beam_merge_with_batch(
                enc_padding_mask, beam_size)
            outputs, _ = dynamic_decode(
                beam_search_decoder,
                inits=dec_initial_states,
                max_step_num=self.beam_max_step_num,
                enc_output=enc_output,
                enc_padding_mask=enc_padding_mask)
            return outputs

        return dec_output
