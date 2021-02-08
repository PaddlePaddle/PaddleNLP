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
from paddle.fluid.contrib.layers import basic_lstm, BasicLSTMUnit

from reader import BOS_ID, EOS_ID

INF = 1. * 1e5
alpha = 0.6
normal_initializer = lambda x: fluid.initializer.NormalInitializer(loc=0., scale=x**-0.5)
uniform_initializer = lambda x: fluid.initializer.UniformInitializer(low=-x, high=x)
zero_constant = fluid.initializer.Constant(0.0)


class EncoderCell(RNNCell):
    def __init__(self,
                 num_layers,
                 hidden_size,
                 param_attr_initializer,
                 param_attr_scale,
                 dropout_prob=0.):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.lstm_cells = []

        for i in range(num_layers):
            lstm_name = "enc_layers_" + str(i)
            self.lstm_cells.append(
                LSTMCell(
                    hidden_size, forget_bias=0., name=lstm_name))

    def call(self, step_input, states):
        new_states = []
        for i in range(self.num_layers):
            out, new_state = self.lstm_cells[i](step_input, states[i])
            step_input = layers.dropout(
                out,
                self.dropout_prob,
                dropout_implementation="upscale_in_train"
            ) if self.dropout_prob > 0 else out
            new_states.append(new_state)
        return step_input, new_states

    @property
    def state_shape(self):
        return [cell.state_shape for cell in self.lstm_cells]


class DecoderCell(RNNCell):
    def __init__(self,
                 num_layers,
                 hidden_size,
                 latent_z,
                 param_attr_initializer,
                 param_attr_scale,
                 dropout_prob=0.):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.latent_z = latent_z
        self.lstm_cells = []

        param_attr = ParamAttr(
            initializer=param_attr_initializer(param_attr_scale))

        for i in range(num_layers):
            lstm_name = "dec_layers_" + str(i)
            self.lstm_cells.append(
                LSTMCell(
                    hidden_size, param_attr, forget_bias=0., name=lstm_name))

    def call(self, step_input, states):
        lstm_states = states
        new_lstm_states = []
        step_input = layers.concat([step_input, self.latent_z], 1)
        for i in range(self.num_layers):
            out, lstm_state = self.lstm_cells[i](step_input, lstm_states[i])
            step_input = layers.dropout(
                out,
                self.dropout_prob,
                dropout_implementation="upscale_in_train"
            ) if self.dropout_prob > 0 else out
            new_lstm_states.append(lstm_state)
        return step_input, new_lstm_states


class VAE(object):
    def __init__(self,
                 hidden_size,
                 latent_size,
                 src_vocab_size,
                 tar_vocab_size,
                 batch_size,
                 num_layers=1,
                 init_scale=0.1,
                 dec_dropout_in=0.5,
                 dec_dropout_out=0.5,
                 enc_dropout_in=0.,
                 enc_dropout_out=0.,
                 word_keep_prob=0.5,
                 batch_first=True,
                 attr_init="normal_initializer"):

        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.src_vocab_size = src_vocab_size
        self.tar_vocab_size = tar_vocab_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.init_scale = init_scale
        self.dec_dropout_in = dec_dropout_in
        self.dec_dropout_out = dec_dropout_out
        self.enc_dropout_in = enc_dropout_in
        self.enc_dropout_out = enc_dropout_out
        self.word_keep_prob = word_keep_prob
        self.batch_first = batch_first

        if attr_init == "normal_initializer":
            self.param_attr_initializer = normal_initializer
            self.param_attr_scale = hidden_size
        elif attr_init == "uniform_initializer":
            self.param_attr_initializer = uniform_initializer
            self.param_attr_scale = init_scale
        else:
            raise TypeError("The type of 'attr_initializer' is not supported")

        self.src_embeder = lambda x: fluid.embedding(
            input=x,
            size=[self.src_vocab_size, self.hidden_size],
            dtype='float32',
            is_sparse=False,
            param_attr=fluid.ParamAttr(
                name='source_embedding',
                initializer=self.param_attr_initializer(self.param_attr_scale)))

        self.tar_embeder = lambda x: fluid.embedding(
            input=x,
            size=[self.tar_vocab_size, self.hidden_size],
            dtype='float32',
            is_sparse=False,
            param_attr=fluid.ParamAttr(
                name='target_embedding',
                initializer=self.param_attr_initializer(self.param_attr_scale)))

    def _build_data(self, mode='train'):
        if mode == 'train':
            self.src = fluid.data(name="src", shape=[None, None], dtype='int64')
            self.src_sequence_length = fluid.data(
                name="src_sequence_length", shape=[None], dtype='int32')
            self.tar = fluid.data(name="tar", shape=[None, None], dtype='int64')
            self.tar_sequence_length = fluid.data(
                name="tar_sequence_length", shape=[None], dtype='int32')
            self.label = fluid.data(
                name="label", shape=[None, None, 1], dtype='int64')
            self.kl_weight = fluid.data(
                name='kl_weight', shape=[1], dtype='float32')
        else:
            self.tar = fluid.data(name="tar", shape=[None, None], dtype='int64')

    def _emebdding(self, mode='train'):
        if mode == 'train':
            self.src_emb = self.src_embeder(self.src)
        self.tar_emb = self.tar_embeder(self.tar)

    def _build_encoder(self):
        self.enc_input = layers.dropout(
            self.src_emb,
            dropout_prob=self.enc_dropout_in,
            dropout_implementation="upscale_in_train")
        enc_cell = EncoderCell(self.num_layers, self.hidden_size,
                               self.param_attr_initializer,
                               self.param_attr_scale, self.enc_dropout_out)
        enc_output, enc_final_state = rnn(
            cell=enc_cell,
            inputs=self.enc_input,
            sequence_length=self.src_sequence_length)
        return enc_output, enc_final_state

    def _build_distribution(self, enc_final_state=None):
        enc_hidden = [
            layers.concat(
                state, axis=-1) for state in enc_final_state
        ]
        enc_hidden = layers.concat(enc_hidden, axis=-1)
        z_mean_log_var = layers.fc(input=enc_hidden,
                                   size=self.latent_size * 2,
                                   name='fc_dist')
        z_mean, z_log_var = layers.split(z_mean_log_var, 2, -1)
        return z_mean, z_log_var

    def _build_decoder(self,
                       z_mean=None,
                       z_log_var=None,
                       enc_output=None,
                       mode='train',
                       beam_size=10):
        dec_input = layers.dropout(
            self.tar_emb,
            dropout_prob=self.dec_dropout_in,
            dropout_implementation="upscale_in_train")

        # `output_layer` will be used within BeamSearchDecoder            
        output_layer = lambda x: layers.fc(x,
                                            size=self.tar_vocab_size,
                                            num_flatten_dims=len(x.shape) - 1,
                                            name="output_w")

        # `sample_output_layer` samples an id from the logits distribution instead of argmax(logits)
        # it will be used within BeamSearchDecoder
        sample_output_layer = lambda x: layers.unsqueeze(fluid.one_hot(
            layers.unsqueeze(
            layers.sampling_id(
                layers.softmax(
                    layers.squeeze(output_layer(x),[1])
                ),dtype='int'), [1]),
                depth=self.tar_vocab_size), [1])

        if mode == 'train':
            latent_z = self._sampling(z_mean, z_log_var)
        else:
            latent_z = layers.gaussian_random_batch_size_like(
                self.tar, shape=[-1, self.latent_size])
        dec_first_hidden_cell = layers.fc(latent_z,
                                          2 * self.hidden_size *
                                          self.num_layers,
                                          name='fc_hc')
        dec_first_hidden, dec_first_cell = layers.split(dec_first_hidden_cell,
                                                        2)
        if self.num_layers > 1:
            dec_first_hidden = layers.split(dec_first_hidden, self.num_layers)
            dec_first_cell = layers.split(dec_first_cell, self.num_layers)
        else:
            dec_first_hidden = [dec_first_hidden]
            dec_first_cell = [dec_first_cell]
        dec_initial_states = [[h, c]
                              for h, c in zip(dec_first_hidden, dec_first_cell)]
        dec_cell = DecoderCell(self.num_layers, self.hidden_size, latent_z,
                               self.param_attr_initializer,
                               self.param_attr_scale, self.dec_dropout_out)

        if mode == 'train':
            dec_output, _ = rnn(cell=dec_cell,
                                inputs=dec_input,
                                initial_states=dec_initial_states,
                                sequence_length=self.tar_sequence_length)
            dec_output = output_layer(dec_output)

            return dec_output
        elif mode == 'greedy':
            start_token = 1
            end_token = 2
            max_length = 100
            beam_search_decoder = BeamSearchDecoder(
                dec_cell,
                start_token,
                end_token,
                beam_size=1,
                embedding_fn=self.tar_embeder,
                output_fn=output_layer)
            outputs, _ = dynamic_decode(
                beam_search_decoder,
                inits=dec_initial_states,
                max_step_num=max_length)
            return outputs

        elif mode == 'sampling':
            start_token = 1
            end_token = 2
            max_length = 100
            beam_search_decoder = BeamSearchDecoder(
                dec_cell,
                start_token,
                end_token,
                beam_size=1,
                embedding_fn=self.tar_embeder,
                output_fn=sample_output_layer)

            outputs, _ = dynamic_decode(
                beam_search_decoder,
                inits=dec_initial_states,
                max_step_num=max_length)
            return outputs
        else:
            print("mode not supprt", mode)

    def _sampling(self, z_mean, z_log_var):
        """reparameterization trick 
        """
        # by default, random_normal has mean=0 and std=1.0
        epsilon = layers.gaussian_random_batch_size_like(
            self.tar, shape=[-1, self.latent_size])
        epsilon.stop_gradient = True
        return z_mean + layers.exp(0.5 * z_log_var) * epsilon

    def _kl_dvg(self, means, logvars):
        """compute the KL divergence between Gaussian distribution
        """
        kl_cost = -0.5 * (logvars - fluid.layers.square(means) -
                          fluid.layers.exp(logvars) + 1.0)
        kl_cost = fluid.layers.reduce_mean(kl_cost, 0)

        return fluid.layers.reduce_sum(kl_cost)

    def _compute_loss(self, mean, logvars, dec_output):

        kl_loss = self._kl_dvg(mean, logvars)

        rec_loss = layers.softmax_with_cross_entropy(
            logits=dec_output, label=self.label, soft_label=False)

        rec_loss = layers.reshape(rec_loss, shape=[self.batch_size, -1])

        max_tar_seq_len = layers.shape(self.tar)[1]
        tar_mask = layers.sequence_mask(
            self.tar_sequence_length, maxlen=max_tar_seq_len, dtype='float32')
        rec_loss = rec_loss * tar_mask
        rec_loss = layers.reduce_mean(rec_loss, dim=[0])
        rec_loss = layers.reduce_sum(rec_loss)

        loss = kl_loss * self.kl_weight + rec_loss

        return loss, kl_loss, rec_loss

    def _beam_search(self, enc_last_hidden, enc_last_cell):
        pass

    def build_graph(self, mode='train', beam_size=10):
        if mode == 'train' or mode == 'eval':
            self._build_data()
            self._emebdding()
            enc_output, enc_final_state = self._build_encoder()
            z_mean, z_log_var = self._build_distribution(enc_final_state)
            dec_output = self._build_decoder(z_mean, z_log_var, enc_output)

            loss, kl_loss, rec_loss = self._compute_loss(z_mean, z_log_var,
                                                         dec_output)
            return loss, kl_loss, rec_loss

        elif mode == "sampling" or mode == 'greedy':
            self._build_data(mode)
            self._emebdding(mode)
            dec_output = self._build_decoder(mode=mode, beam_size=1)

            return dec_output
        else:
            print("not support mode ", mode)
            raise Exception("not support mode: " + mode)
