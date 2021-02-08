#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""
MMDNN class
"""
import numpy as np
import paddle.fluid as fluid
import logging


class MMDNN(object):
    """
    MMDNN
    """

    def __init__(self, config):
        """
        initialize
        """
        self.vocab_size = int(config['dict_size'])
        self.emb_size = int(config['net']['embedding_dim'])
        self.lstm_dim = int(config['net']['lstm_dim'])
        self.kernel_size = int(config['net']['num_filters'])
        self.win_size1 = int(config['net']['window_size_left'])
        self.win_size2 = int(config['net']['window_size_right'])
        self.dpool_size1 = int(config['net']['dpool_size_left'])
        self.dpool_size2 = int(config['net']['dpool_size_right'])
        self.hidden_size = int(config['net']['hidden_size'])
        self.seq_len1 = int(config['max_len_left'])
        self.seq_len2 = int(config['max_len_right'])
        self.task_mode = config['task_mode']

        if int(config['match_mask']) != 0:
            self.match_mask = True
        else:
            self.match_mask = False

        if self.task_mode == "pointwise":
            self.n_class = int(config['n_class'])
            self.out_size = self.n_class
        elif self.task_mode == "pairwise":
            self.out_size = 1
        else:
            logging.error("training mode not supported")

    def embedding_layer(self, input, zero_pad=True, scale=True):
        """
        embedding layer
        """
        emb = fluid.layers.embedding(
            input=input,
            size=[self.vocab_size, self.emb_size],
            padding_idx=(0 if zero_pad else None),
            param_attr=fluid.ParamAttr(
                name="word_embedding", initializer=fluid.initializer.Xavier()))
        if scale:
            emb = emb * (self.emb_size**0.5)
        return emb

    def bi_dynamic_lstm(self, input, hidden_size):
        """
        bi_lstm layer
        """
        fw_in_proj = fluid.layers.fc(input=input,
                                     size=4 * hidden_size,
                                     param_attr=fluid.ParamAttr(name="fw_fc.w"),
                                     bias_attr=False)
        forward, _ = fluid.layers.dynamic_lstm(
            input=fw_in_proj,
            size=4 * hidden_size,
            is_reverse=False,
            param_attr=fluid.ParamAttr(name="forward_lstm.w"),
            bias_attr=fluid.ParamAttr(name="forward_lstm.b"))

        rv_in_proj = fluid.layers.fc(input=input,
                                     size=4 * hidden_size,
                                     param_attr=fluid.ParamAttr(name="rv_fc.w"),
                                     bias_attr=False)
        reverse, _ = fluid.layers.dynamic_lstm(
            input=rv_in_proj,
            size=4 * hidden_size,
            is_reverse=True,
            param_attr=fluid.ParamAttr(name="reverse_lstm.w"),
            bias_attr=fluid.ParamAttr(name="reverse_lstm.b"))
        return [forward, reverse]

    def conv_pool_relu_layer(self, input, mask=None):
        """
        convolution and pool layer
        """
        # data format NCHW
        emb_expanded = fluid.layers.unsqueeze(input=input, axes=[1])
        # same padding
        conv = fluid.layers.conv2d(
            input=emb_expanded,
            num_filters=self.kernel_size,
            stride=1,
            padding=(int(self.seq_len1 / 2), int(self.seq_len2 // 2)),
            filter_size=(self.seq_len1, self.seq_len2),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(0.1)))

        if mask is not None:
            cross_mask = fluid.layers.stack(x=[mask] * self.kernel_size, axis=1)
            conv = cross_mask * conv + (1 - cross_mask) * (-2**32 + 1)
        # valid padding
        pool = fluid.layers.pool2d(
            input=conv,
            pool_size=[
                int(self.seq_len1 / self.dpool_size1),
                int(self.seq_len2 / self.dpool_size2)
            ],
            pool_stride=[
                int(self.seq_len1 / self.dpool_size1),
                int(self.seq_len2 / self.dpool_size2)
            ],
            pool_type="max", )

        relu = fluid.layers.relu(pool)
        return relu

    def get_cross_mask(self, left_lens, right_lens):
        """
        cross mask
        """
        mask1 = fluid.layers.sequence_mask(
            x=left_lens, dtype='float32', maxlen=self.seq_len1 + 1)
        mask2 = fluid.layers.sequence_mask(
            x=right_lens, dtype='float32', maxlen=self.seq_len2 + 1)

        mask1 = fluid.layers.transpose(x=mask1, perm=[0, 2, 1])
        cross_mask = fluid.layers.matmul(x=mask1, y=mask2)
        return cross_mask

    def predict(self, left, right):
        """
        Forward network
        """
        left_emb = self.embedding_layer(left, zero_pad=True, scale=False)
        right_emb = self.embedding_layer(right, zero_pad=True, scale=False)

        bi_left_outputs = self.bi_dynamic_lstm(
            input=left_emb, hidden_size=self.lstm_dim)
        left_seq_encoder = fluid.layers.concat(input=bi_left_outputs, axis=1)
        bi_right_outputs = self.bi_dynamic_lstm(
            input=right_emb, hidden_size=self.lstm_dim)
        right_seq_encoder = fluid.layers.concat(input=bi_right_outputs, axis=1)

        pad_value = fluid.layers.assign(input=np.array([0]).astype("float32"))
        left_seq_encoder, left_lens = fluid.layers.sequence_pad(
            x=left_seq_encoder, pad_value=pad_value, maxlen=self.seq_len1)
        right_seq_encoder, right_lens = fluid.layers.sequence_pad(
            x=right_seq_encoder, pad_value=pad_value, maxlen=self.seq_len2)

        cross = fluid.layers.matmul(
            left_seq_encoder, right_seq_encoder, transpose_y=True)
        if self.match_mask:
            cross_mask = self.get_cross_mask(left_lens, right_lens)
        else:
            cross_mask = None
        conv_pool_relu = self.conv_pool_relu_layer(input=cross, mask=cross_mask)
        relu_hid1 = fluid.layers.fc(input=conv_pool_relu,
                                    act="tanh",
                                    size=self.hidden_size)

        pred = fluid.layers.fc(input=relu_hid1,
                               size=self.out_size,
                               act="softmax")

        return left_seq_encoder, pred
