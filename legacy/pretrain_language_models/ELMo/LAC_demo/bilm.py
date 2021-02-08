#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
import os

cell_clip = 3.0
proj_clip = 3.0
hidden_size = 4096
vocab_size = 52445
emb_size = 512


def dropout(input):
    dropout_rate = 0.5
    return layers.dropout(
        input,
        dropout_prob=dropout_rate,
        dropout_implementation="upscale_in_train",
        is_test=False)


def lstmp_encoder(input_seq, gate_size, h_0, c_0, para_name, proj_size, args):
    # A lstm encoder implementation with projection.
    # Linear transformation part for input gate, output gate, forget gate
    # and cell activation vectors need be done outside of dynamic_lstm.
    # So the output size is 4 times of gate_size.

    init = None
    init_b = None
    input_proj = layers.fc(input=input_seq,
                           param_attr=fluid.ParamAttr(
                               name=para_name + '_gate_w', initializer=init),
                           size=gate_size * 4,
                           act=None,
                           bias_attr=False)
    hidden, cell = layers.dynamic_lstmp(
        input=input_proj,
        size=gate_size * 4,
        proj_size=proj_size,
        h_0=h_0,
        c_0=c_0,
        use_peepholes=False,
        proj_clip=proj_clip,
        cell_clip=cell_clip,
        proj_activation="identity",
        param_attr=fluid.ParamAttr(initializer=init),
        bias_attr=fluid.ParamAttr(initializer=init_b))
    return hidden, cell, input_proj


def encoder_wrapper(x_emb,
                    vocab_size,
                    emb_size,
                    init_hidden=None,
                    init_cell=None,
                    para_name='',
                    args=None):
    rnn_input = x_emb
    rnn_outs = []
    rnn_outs_ori = []
    cells = []
    projs = []
    num_layers = 2
    for i in range(num_layers):
        if init_hidden and init_cell:
            h0 = layers.squeeze(
                layers.slice(
                    init_hidden, axes=[0], starts=[i], ends=[i + 1]),
                axes=[0])
            c0 = layers.squeeze(
                layers.slice(
                    init_cell, axes=[0], starts=[i], ends=[i + 1]),
                axes=[0])
        else:
            h0 = c0 = None
        rnn_out, cell, input_proj = lstmp_encoder(
            rnn_input, hidden_size, h0, c0, para_name + 'layer{}'.format(i + 1),
            emb_size, args)
        rnn_out_ori = rnn_out
        if i > 0:
            rnn_out = rnn_out + rnn_input
        rnn_out.stop_gradient = True
        rnn_outs.append(rnn_out)
        rnn_outs_ori.append(rnn_out_ori)
    return rnn_outs, rnn_outs_ori


def weight_layers(lm_embeddings, name="", l2_coef=0.0):
    '''
    Weight the layers of a biLM with trainable scalar weights to
    compute ELMo representations.

    Input:
        lm_embeddings(list): representations of 2 layers from biLM.
        name = a string prefix used for the trainable variable names
        l2_coef: the l2 regularization coefficient $\lambda$.
            Pass None or 0.0 for no regularization.

    Output:
        weighted_lm_layers: weighted embeddings form biLM
    '''

    n_lm_layers = len(lm_embeddings)
    W = layers.create_parameter(
        [n_lm_layers, ],
        dtype="float32",
        name=name + "ELMo_w",
        attr=fluid.ParamAttr(
            name=name + "ELMo_w",
            initializer=fluid.initializer.Constant(0.0),
            regularizer=fluid.regularizer.L2Decay(l2_coef)))
    normed_weights = layers.softmax(W + 1.0 / n_lm_layers)
    splited_normed_weights = layers.split(normed_weights, n_lm_layers, dim=0)

    # compute the weighted, normalized LM activations
    pieces = []
    for w, t in zip(splited_normed_weights, lm_embeddings):
        pieces.append(t * w)
    sum_pieces = layers.sums(pieces)

    # scale the weighted sum by gamma
    gamma = layers.create_parameter(
        [1],
        dtype="float32",
        name=name + "ELMo_gamma",
        attr=fluid.ParamAttr(
            name=name + "ELMo_gamma",
            initializer=fluid.initializer.Constant(1.0)))
    weighted_lm_layers = sum_pieces * gamma
    return weighted_lm_layers


def elmo_encoder(word_ids, elmo_l2_coef):

    x_emb = layers.embedding(
        input=word_ids,
        size=[vocab_size, emb_size],
        dtype='float32',
        is_sparse=False,
        param_attr=fluid.ParamAttr(name='embedding_para'))

    x_emb_r = fluid.layers.sequence_reverse(x_emb, name=None)
    fw_hiddens, fw_hiddens_ori = encoder_wrapper(
        x_emb, vocab_size, emb_size, para_name='fw_', args=None)
    bw_hiddens, bw_hiddens_ori = encoder_wrapper(
        x_emb_r, vocab_size, emb_size, para_name='bw_', args=None)

    num_layers = len(fw_hiddens_ori)
    token_embeddings = layers.concat(input=[x_emb, x_emb], axis=1)
    token_embeddings.stop_gradient = True
    concate_embeddings = [token_embeddings]
    for index in range(num_layers):
        embedding = layers.concat(
            input=[fw_hiddens_ori[index], bw_hiddens_ori[index]], axis=1)
        embedding = dropout(embedding)
        embedding.stop_gradient = True
        concate_embeddings.append(embedding)
    weighted_emb = weight_layers(concate_embeddings, l2_coef=elmo_l2_coef)
    return weighted_emb


def init_pretraining_params(exe, pretraining_params_path, main_program):
    assert os.path.exists(pretraining_params_path
                          ), "[%s] cann't be found." % pretraining_params_path

    def if_exist(var):
        path = os.path.join(pretraining_params_path, var.name)
        exist = os.path.exists(path)
        if exist:
            print('Load model: %s' % path)
        return exist

    fluid.io.load_vars(
        exe,
        pretraining_params_path,
        main_program=main_program,
        predicate=if_exist)
    print("Load pretraining parameters from {}.".format(
        pretraining_params_path))
