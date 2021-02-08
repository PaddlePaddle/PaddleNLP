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


def dropout(input, test_mode, args):
    if args.dropout and (not test_mode):
        return layers.dropout(
            input,
            dropout_prob=args.dropout,
            dropout_implementation="upscale_in_train",
            seed=args.random_seed,
            is_test=False)
    else:
        return input


def lstmp_encoder(input_seq, gate_size, h_0, c_0, para_name, proj_size,
                  test_mode, args):
    # A lstm encoder implementation with projection.
    # Linear transformation part for input gate, output gate, forget gate
    # and cell activation vectors need be done outside of dynamic_lstm.
    # So the output size is 4 times of gate_size.

    input_seq = dropout(input_seq, test_mode, args)
    input_proj = layers.fc(input=input_seq,
                           param_attr=fluid.ParamAttr(
                               name=para_name + '_gate_w', initializer=None),
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
        proj_clip=args.proj_clip,
        cell_clip=args.cell_clip,
        proj_activation="identity",
        param_attr=fluid.ParamAttr(initializer=None),
        bias_attr=fluid.ParamAttr(initializer=None))

    return hidden, cell, input_proj


def encoder(x,
            y,
            vocab_size,
            emb_size,
            init_hidden=None,
            init_cell=None,
            para_name='',
            custom_samples=None,
            custom_probabilities=None,
            test_mode=False,
            args=None):
    x_emb = layers.embedding(
        input=x,
        size=[vocab_size, emb_size],
        dtype='float32',
        is_sparse=False,
        param_attr=fluid.ParamAttr(name='embedding_para'))
    rnn_input = x_emb
    rnn_outs = []
    rnn_outs_ori = []
    cells = []
    projs = []
    for i in range(args.num_layers):
        rnn_input = dropout(rnn_input, test_mode, args)
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
            rnn_input, args.hidden_size, h0, c0,
            para_name + 'layer{}'.format(i + 1), emb_size, test_mode, args)
        rnn_out_ori = rnn_out
        if i > 0:
            rnn_out = rnn_out + rnn_input
        rnn_out = dropout(rnn_out, test_mode, args)
        cell = dropout(cell, test_mode, args)
        rnn_outs.append(rnn_out)
        rnn_outs_ori.append(rnn_out_ori)
        rnn_input = rnn_out
        cells.append(cell)
        projs.append(input_proj)

    softmax_weight = layers.create_parameter(
        [vocab_size, emb_size], dtype="float32", name="softmax_weight")
    softmax_bias = layers.create_parameter(
        [vocab_size], dtype="float32", name='softmax_bias')
    projection = layers.matmul(rnn_outs[-1], softmax_weight, transpose_y=True)
    projection = layers.elementwise_add(projection, softmax_bias)

    projection = layers.reshape(projection, shape=[-1, vocab_size])

    if args.sample_softmax and (not test_mode):
        loss = layers.sampled_softmax_with_cross_entropy(
            logits=projection,
            label=y,
            num_samples=args.n_negative_samples_batch,
            seed=args.random_seed)
    else:
        label = layers.one_hot(input=y, depth=vocab_size)
        loss = layers.softmax_with_cross_entropy(
            logits=projection, label=label, soft_label=True)
    return [x_emb, projection, loss], rnn_outs, rnn_outs_ori, cells, projs


class LanguageModel(object):
    def __init__(self, args, vocab_size, test_mode):
        self.args = args
        self.vocab_size = vocab_size
        self.test_mode = test_mode

    def build(self):
        args = self.args
        emb_size = args.embed_size
        proj_size = args.embed_size
        hidden_size = args.hidden_size
        batch_size = args.batch_size
        num_layers = args.num_layers
        num_steps = args.num_steps

        lstm_outputs = []

        x_f = layers.data(name="x", shape=[1], dtype='int64', lod_level=1)
        y_f = layers.data(name="y", shape=[1], dtype='int64', lod_level=1)

        x_b = layers.data(name="x_r", shape=[1], dtype='int64', lod_level=1)
        y_b = layers.data(name="y_r", shape=[1], dtype='int64', lod_level=1)

        init_hiddens_ = layers.data(
            name="init_hiddens", shape=[1], dtype='float32')
        init_cells_ = layers.data(name="init_cells", shape=[1], dtype='float32')

        init_hiddens = layers.reshape(
            init_hiddens_, shape=[2 * num_layers, -1, proj_size])
        init_cells = layers.reshape(
            init_cells_, shape=[2 * num_layers, -1, hidden_size])

        init_hidden = layers.slice(
            init_hiddens, axes=[0], starts=[0], ends=[num_layers])
        init_cell = layers.slice(
            init_cells, axes=[0], starts=[0], ends=[num_layers])
        init_hidden_r = layers.slice(
            init_hiddens, axes=[0], starts=[num_layers], ends=[2 * num_layers])
        init_cell_r = layers.slice(
            init_cells, axes=[0], starts=[num_layers], ends=[2 * num_layers])

        if args.use_custom_samples:
            custom_samples = layers.data(
                name="custom_samples",
                shape=[args.n_negative_samples_batch + 1],
                dtype='int64',
                lod_level=1)
            custom_samples_r = layers.data(
                name="custom_samples_r",
                shape=[args.n_negative_samples_batch + 1],
                dtype='int64',
                lod_level=1)
            custom_probabilities = layers.data(
                name="custom_probabilities",
                shape=[args.n_negative_samples_batch + 1],
                dtype='float32',
                lod_level=1)
        else:
            custom_samples = None
            custom_samples_r = None
            custom_probabilities = None

        forward, fw_hiddens, fw_hiddens_ori, fw_cells, fw_projs = encoder(
            x_f,
            y_f,
            self.vocab_size,
            emb_size,
            init_hidden,
            init_cell,
            para_name='fw_',
            custom_samples=custom_samples,
            custom_probabilities=custom_probabilities,
            test_mode=self.test_mode,
            args=args)
        backward, bw_hiddens, bw_hiddens_ori, bw_cells, bw_projs = encoder(
            x_b,
            y_b,
            self.vocab_size,
            emb_size,
            init_hidden_r,
            init_cell_r,
            para_name='bw_',
            custom_samples=custom_samples_r,
            custom_probabilities=custom_probabilities,
            test_mode=self.test_mode,
            args=args)

        losses = layers.concat([forward[-1], backward[-1]])
        self.loss = layers.reduce_mean(losses)
        self.loss.persistable = True
        self.grad_vars = [x_f, y_f, x_b, y_b, self.loss]
        self.grad_vars_name = ['x', 'y', 'x_r', 'y_r', 'final_loss']
        fw_vars_name = ['x_emb', 'proj', 'loss'] + [
            'init_hidden', 'init_cell'
        ] + ['rnn_out', 'rnn_out2', 'cell', 'cell2', 'xproj', 'xproj2']
        bw_vars_name = ['x_emb_r', 'proj_r', 'loss_r'] + [
            'init_hidden_r', 'init_cell_r'
        ] + [
            'rnn_out_r', 'rnn_out2_r', 'cell_r', 'cell2_r', 'xproj_r',
            'xproj2_r'
        ]
        fw_vars = forward + [init_hidden, init_cell
                             ] + fw_hiddens + fw_cells + fw_projs
        bw_vars = backward + [init_hidden_r, init_cell_r
                              ] + bw_hiddens + bw_cells + bw_projs
        for i in range(len(fw_vars_name)):
            self.grad_vars.append(fw_vars[i])
            self.grad_vars.append(bw_vars[i])
            self.grad_vars_name.append(fw_vars_name[i])
            self.grad_vars_name.append(bw_vars_name[i])
        if args.use_custom_samples:
            self.feed_order = [
                'x', 'y', 'x_r', 'y_r', 'custom_samples', 'custom_samples_r',
                'custom_probabilities'
            ]
        else:
            self.feed_order = ['x', 'y', 'x_r', 'y_r']
        self.last_hidden = [
            fluid.layers.sequence_last_step(input=x)
            for x in fw_hiddens_ori + bw_hiddens_ori
        ]
        self.last_cell = [
            fluid.layers.sequence_last_step(input=x)
            for x in fw_cells + bw_cells
        ]
        self.last_hidden = layers.concat(self.last_hidden, axis=0)
        self.last_hidden.persistable = True
        self.last_cell = layers.concat(self.last_cell, axis=0)
        self.last_cell.persistable = True
