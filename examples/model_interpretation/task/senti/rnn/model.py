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

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddlenlp as nlp

INF = 1. * 1e12


class LSTMModel(nn.Layer):

    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 lstm_hidden_size=198,
                 direction='forward',
                 lstm_layers=1,
                 dropout_rate=0.0,
                 pooling_type=None,
                 fc_hidden_size=96):
        super().__init__()

        self.direction = direction

        self.embedder = nn.Embedding(num_embeddings=vocab_size,
                                     embedding_dim=emb_dim,
                                     padding_idx=padding_idx)

        # self.lstm_encoder = nlp.seq2vec.LSTMEncoder(emb_dim,
        #                                             lstm_hidden_size,
        #                                             num_layers=lstm_layers,
        #                                             direction=direction,
        #                                             dropout=dropout_rate,
        #                                             pooling_type=pooling_type)

        self.lstm_layer = nn.LSTM(input_size=emb_dim,
                                  hidden_size=lstm_hidden_size,
                                  num_layers=lstm_layers,
                                  direction=direction,
                                  dropout=dropout_rate)

        self.fc = nn.Linear(
            lstm_hidden_size * (2 if direction == 'bidirect' else 1),
            fc_hidden_size)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)
        self.softmax = nn.Softmax(axis=1)

    def forward(self, text, seq_len):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens, num_directions*lstm_hidden_size)
        # num_directions = 2 if direction is 'bidirect'
        # if not, num_directions = 1

        # text_repr = self.lstm_encoder(embedded_text, sequence_length=seq_len)

        encoded_text, (last_hidden,
                       last_cell) = self.lstm_layer(embedded_text,
                                                    sequence_length=seq_len)
        if self.direction == 'bidirect':
            text_repr = paddle.concat(
                (last_hidden[-2, :, :], last_hidden[-1, :, :]), axis=1)
        else:
            text_repr = last_hidden[-1, :, :]

        fc_out = paddle.tanh(
            self.fc(text_repr))  # Shape: (batch_size, fc_hidden_size)
        logits = self.output_layer(fc_out)  # Shape: (batch_size, num_classes)
        return logits

    def forward_interpet(self, text, seq_len):
        embedded_text = self.embedder(
            text)  # Shape: (batch_size, num_tokens, embedding_dim)

        # text_repr = self.lstm_encoder(embedded_text, sequence_length=seq_len) # Shape: (batch_size, num_tokens, num_directions * hidden)

        # encoded_text: tensor[batch, seq_len, num_directions * hidden]
        # last_hidden: tensor[2, batch, hiddens]
        encoded_text, (last_hidden,
                       last_cell) = self.lstm_layer(embedded_text,
                                                    sequence_length=seq_len)
        if self.direction == 'bidirect':
            text_repr = paddle.concat(
                (last_hidden[-2, :, :], last_hidden[-1, :, :]),
                axis=1)  # text_repr: tensor[batch, 2 * hidden] 双向
        else:
            text_repr = last_hidden[
                -1, :, :]  # text_repr: tensor[1, hidden_size]     单向

        fc_out = paddle.tanh(
            self.fc(text_repr))  # Shape: (batch_size, fc_hidden_size)
        logits = self.output_layer(fc_out)  # Shape: (batch_size, num_classes)
        probs = self.softmax(logits)

        return probs, text_repr, embedded_text


class BiLSTMAttentionModel(nn.Layer):

    def __init__(self,
                 attention_layer,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 lstm_hidden_size=196,
                 fc_hidden_size=96,
                 lstm_layers=1,
                 dropout_rate=0.0,
                 padding_idx=0):
        super().__init__()
        self.padding_idx = padding_idx

        self.embedder = nn.Embedding(num_embeddings=vocab_size,
                                     embedding_dim=emb_dim,
                                     padding_idx=padding_idx)
        self.bilstm = nn.LSTM(input_size=emb_dim,
                              hidden_size=lstm_hidden_size,
                              num_layers=lstm_layers,
                              dropout=dropout_rate,
                              direction='bidirect')
        self.attention = attention_layer
        if isinstance(attention_layer, SelfAttention):
            self.fc = nn.Linear(lstm_hidden_size, fc_hidden_size)
        elif isinstance(attention_layer, SelfInteractiveAttention):
            self.fc = nn.Linear(lstm_hidden_size * 2, fc_hidden_size)
        else:
            raise RuntimeError("Unknown attention type %s." %
                               attention_layer.__class__.__name__)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)
        self.softmax = nn.Softmax(axis=1)

    def forward(self, text, seq_len):
        mask = text != self.padding_idx
        embedded_text = self.embedder(text)
        # Encode text, shape: (batch, max_seq_len, num_directions * hidden_size)
        encoded_text, (last_hidden,
                       last_cell) = self.bilstm(embedded_text,
                                                sequence_length=seq_len)
        # Shape: (batch_size, lstm_hidden_size)
        hidden, att_weights = self.attention(
            encoded_text, mask)  # Shape: (batch_size, fc_hidden_size)
        fc_out = paddle.tanh(
            self.fc(hidden))  # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc_out)
        return logits

    def forward_interpet(self,
                         text,
                         seq_len,
                         noise=None,
                         i=None,
                         n_samples=None):
        mask = text != self.padding_idx

        baseline_text = paddle.to_tensor([[0] * text.shape[1]],
                                         dtype=text.dtype,
                                         place=text.place,
                                         stop_gradient=text.stop_gradient)

        embedded_text = self.embedder(text)
        baseline_embedded = self.embedder(baseline_text)

        if noise is not None:
            if noise.upper() == 'GAUSSIAN':
                stdev_spread = 0.15
                stdev = stdev_spread * (embedded_text.max() -
                                        embedded_text.min()).numpy()
                noise = paddle.to_tensor(np.random.normal(
                    0, stdev, embedded_text.shape).astype(np.float32),
                                         stop_gradient=False)
                embedded_text = embedded_text + noise

            elif noise.upper() == 'INTEGRATED':
                embedded_text = baseline_embedded + (i / (n_samples - 1)) * (
                    embedded_text - baseline_embedded)

            else:
                raise ValueError('unsupported noise method: %s' % (noise))

        # Encode text, shape: (batch, max_seq_len, num_directions * hidden_size)
        encoded_text, (last_hidden,
                       last_cell) = self.bilstm(embedded_text,
                                                sequence_length=seq_len)
        # Shape: (batch_size, lstm_hidden_size)
        hidden, att_weights = self.attention(
            encoded_text, mask)  # Shape: (batch_size, fc_hidden_size)
        fc_out = paddle.tanh(
            self.fc(hidden))  # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc_out)
        probs = self.softmax(logits)
        return probs, att_weights.squeeze(axis=-1), embedded_text


class SelfAttention(nn.Layer):
    """
    A close implementation of attention network of ACL 2016 paper, 
    Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification (Zhou et al., 2016).
    ref: https://www.aclweb.org/anthology/P16-2034/
    Args:
        hidden_size (int): The number of expected features in the input x.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.att_weight = self.create_parameter(shape=[1, hidden_size, 1],
                                                dtype='float32')

    def forward(self, input, mask=None):
        """
        Args:
            input (paddle.Tensor) of shape (batch, seq_len, input_size): Tensor containing the features of the input sequence.
            mask (paddle.Tensor) of shape (batch, seq_len) :
                Tensor is a bool tensor, whose each element identifies whether the input word id is pad token or not. 
                Defaults to `None`.
        """
        forward_input, backward_input = paddle.chunk(input, chunks=2, axis=2)
        # elementwise-sum forward_x and backward_x
        # Shape: (batch_size, max_seq_len, hidden_size)
        h = paddle.add_n([forward_input, backward_input])
        # Shape: (batch_size, hidden_size, 1)
        att_weight = self.att_weight.tile(repeat_times=(paddle.shape(h)[0], 1,
                                                        1))
        # Shape: (batch_size, max_seq_len, 1)
        att_score = paddle.bmm(paddle.tanh(h), att_weight)
        if mask is not None:
            # mask, remove the effect of 'PAD'
            mask = paddle.cast(mask, dtype='float32')
            mask = mask.unsqueeze(axis=-1)
            inf_tensor = paddle.full(shape=mask.shape,
                                     dtype='float32',
                                     fill_value=-INF)
            att_score = paddle.multiply(att_score, mask) + paddle.multiply(
                inf_tensor, (1 - mask))
        # Shape: (batch_size, max_seq_len, 1)
        att_weight = F.softmax(att_score, axis=1)
        # Shape: (batch_size, lstm_hidden_size)
        reps = paddle.bmm(h.transpose(perm=(0, 2, 1)),
                          att_weight).squeeze(axis=-1)
        reps = paddle.tanh(reps)
        return reps, att_weight


class SelfInteractiveAttention(nn.Layer):
    """
    A close implementation of attention network of NAACL 2016 paper, Hierarchical Attention Networks for Document Classiﬁcation (Yang et al., 2016).
    ref: https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf
    Args:
        hidden_size (int): The number of expected features in the input x.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.input_weight = self.create_parameter(
            shape=[1, hidden_size, hidden_size], dtype='float32')
        self.bias = self.create_parameter(shape=[1, 1, hidden_size],
                                          dtype='float32')
        self.att_context_vector = self.create_parameter(
            shape=[1, hidden_size, 1], dtype='float32')

    def forward(self, input, mask=None):
        """
        Args:
            input (paddle.Tensor) of shape (batch, seq_len, hidden_size): Tensor containing the features of the input sequence.
            mask (paddle.Tensor) of shape (batch, seq_len) :
                Tensor is a bool tensor, whose each element identifies whether the input word id is pad token or not.
                Defaults to `None
        """
        weight = self.input_weight.tile(
            repeat_times=(paddle.shape(input)[0], 1,
                          1))  # tensor[batch, hidden_size, hidden_size]
        bias = self.bias.tile(repeat_times=(paddle.shape(input)[0], 1,
                                            1))  # tensor[batch, 1, hidden_size]
        word_squish = paddle.bmm(
            input, weight) + bias  # Shape: (batch_size, seq_len, hidden_size)
        att_context_vector = self.att_context_vector.tile(
            repeat_times=(paddle.shape(input)[0], 1,
                          1))  # Shape: (batch_size, hidden_size, 1)
        att_score = paddle.bmm(
            word_squish, att_context_vector)  # tensor[batch_size, seq_len, 1]
        if mask is not None:
            # mask, remove the effect of 'PAD'
            mask = paddle.cast(mask, dtype='float32')
            mask = mask.unsqueeze(axis=-1)
            inf_tensor = paddle.full(shape=paddle.shape(mask),
                                     dtype='float32',
                                     fill_value=-INF)
            att_score = paddle.multiply(att_score, mask) + paddle.multiply(
                inf_tensor, (1 - mask))
        att_weight = F.softmax(att_score,
                               axis=1)  # tensor[batch_size, seq_len, 1]

        reps = paddle.bmm(input.transpose(perm=(0, 2, 1)), att_weight).squeeze(
            -1)  # Shape: (batch_size, hidden_size)
        return reps, att_weight
