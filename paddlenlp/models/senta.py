# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddlenlp as nlp

INF = 1. * 1e12


class Senta(nn.Layer):
    def __init__(self,
                 network,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 pad_token_id=0):
        super().__init__()

        network = network.lower()
        if network == 'bow':
            self.model = BoWModel(
                vocab_size, num_classes, emb_dim, padding_idx=pad_token_id)
        elif network == 'bigru':
            self.model = GRUModel(
                vocab_size,
                num_classes,
                emb_dim,
                direction='bidirect',
                padding_idx=pad_token_id)
        elif network == 'bilstm':
            self.model = LSTMModel(
                vocab_size,
                num_classes,
                emb_dim,
                direction='bidirect',
                padding_idx=pad_token_id)
        elif network == 'bilstm_attn':
            lstm_hidden_size = 196
            attention = SelfInteractiveAttention(hidden_size=2 *
                                                 lstm_hidden_size)
            self.model = BiLSTMAttentionModel(
                attention_layer=attention,
                vocab_size=vocab_size,
                lstm_hidden_size=lstm_hidden_size,
                num_classes=num_classes,
                padding_idx=pad_token_id)
        elif network == 'birnn':
            self.model = RNNModel(
                vocab_size,
                num_classes,
                emb_dim,
                direction='bidirect',
                padding_idx=pad_token_id)
        elif network == 'cnn':
            self.model = CNNModel(
                vocab_size, num_classes, emb_dim, padding_idx=pad_token_id)
        elif network == 'gru':
            self.model = GRUModel(
                vocab_size,
                num_classes,
                emb_dim,
                direction='forward',
                padding_idx=pad_token_id,
                pooling_type='max')
        elif network == 'lstm':
            self.model = LSTMModel(
                vocab_size,
                num_classes,
                emb_dim,
                direction='forward',
                padding_idx=pad_token_id,
                pooling_type='max')
        elif network == 'rnn':
            self.model = RNNModel(
                vocab_size,
                num_classes,
                emb_dim,
                direction='forward',
                padding_idx=pad_token_id,
                pooling_type='max')
        elif network == 'textcnn':
            self.model = TextCNNModel(
                vocab_size, num_classes, emb_dim, padding_idx=pad_token_id)
        else:
            raise ValueError(
                "Unknown network: %s, it must be one of bow, lstm, bilstm, cnn, gru, bigru, rnn, birnn, bilstm_attn and textcnn."
                % network)

    def forward(self, text, seq_len=None):
        logits = self.model(text, seq_len)
        return logits


class BoWModel(nn.Layer):
    """
    This class implements the Bag of Words Classification Network model to classify texts.
    At a high level, the model starts by embedding the tokens and running them through
    a word embedding. Then, we encode these epresentations with a `BoWEncoder`.
    Lastly, we take the output of the encoder to create a final representation,
    which is passed through some feed-forward layers to output a logits (`output_layer`).

    """

    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 hidden_size=128,
                 fc_hidden_size=96):
        super().__init__()
        self.embedder = nn.Embedding(
            vocab_size, emb_dim, padding_idx=padding_idx)
        self.bow_encoder = nlp.seq2vec.BoWEncoder(emb_dim)
        self.fc1 = nn.Linear(self.bow_encoder.get_output_dim(), hidden_size)
        self.fc2 = nn.Linear(hidden_size, fc_hidden_size)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, text, seq_len=None):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)

        # Shape: (batch_size, embedding_dim)
        summed = self.bow_encoder(embedded_text)
        encoded_text = paddle.tanh(summed)

        # Shape: (batch_size, hidden_size)
        fc1_out = paddle.tanh(self.fc1(encoded_text))
        # Shape: (batch_size, fc_hidden_size)
        fc2_out = paddle.tanh(self.fc2(fc1_out))
        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc2_out)
        return logits


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
        self.embedder = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=padding_idx)
        self.lstm_encoder = nlp.seq2vec.LSTMEncoder(
            emb_dim,
            lstm_hidden_size,
            num_layers=lstm_layers,
            direction=direction,
            dropout=dropout_rate,
            pooling_type=pooling_type)
        self.fc = nn.Linear(self.lstm_encoder.get_output_dim(), fc_hidden_size)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, text, seq_len):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens, num_directions*lstm_hidden_size)
        # num_directions = 2 if direction is 'bidirect'
        # if not, num_directions = 1
        text_repr = self.lstm_encoder(embedded_text, sequence_length=seq_len)
        # Shape: (batch_size, fc_hidden_size)
        fc_out = paddle.tanh(self.fc(text_repr))
        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc_out)
        return logits


class GRUModel(nn.Layer):
    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 gru_hidden_size=198,
                 direction='forward',
                 gru_layers=1,
                 dropout_rate=0.0,
                 pooling_type=None,
                 fc_hidden_size=96):
        super().__init__()
        self.embedder = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=padding_idx)
        self.gru_encoder = nlp.seq2vec.GRUEncoder(
            emb_dim,
            gru_hidden_size,
            num_layers=gru_layers,
            direction=direction,
            dropout=dropout_rate,
            pooling_type=pooling_type)
        self.fc = nn.Linear(self.gru_encoder.get_output_dim(), fc_hidden_size)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, text, seq_len):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens, num_directions*gru_hidden_size)
        # num_directions = 2 if direction is 'bidirect'
        # if not, num_directions = 1
        text_repr = self.gru_encoder(embedded_text, sequence_length=seq_len)
        # Shape: (batch_size, fc_hidden_size)
        fc_out = paddle.tanh(self.fc(text_repr))
        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc_out)
        return logits


class RNNModel(nn.Layer):
    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 rnn_hidden_size=198,
                 direction='forward',
                 rnn_layers=1,
                 dropout_rate=0.0,
                 pooling_type=None,
                 fc_hidden_size=96):
        super().__init__()
        self.embedder = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=padding_idx)
        self.rnn_encoder = nlp.seq2vec.RNNEncoder(
            emb_dim,
            rnn_hidden_size,
            num_layers=rnn_layers,
            direction=direction,
            dropout=dropout_rate,
            pooling_type=pooling_type)
        self.fc = nn.Linear(self.rnn_encoder.get_output_dim(), fc_hidden_size)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, text, seq_len):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens, num_directions*rnn_hidden_size)
        # num_directions = 2 if direction is 'bidirect'
        # if not, num_directions = 1
        text_repr = self.rnn_encoder(embedded_text, sequence_length=seq_len)
        # Shape: (batch_size, fc_hidden_size)
        fc_out = paddle.tanh(self.fc(text_repr))
        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc_out)
        return logits


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

        self.embedder = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=padding_idx)
        self.bilstm = nn.LSTM(
            input_size=emb_dim,
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

    def forward(self, text, seq_len):
        mask = text != self.padding_idx
        embedded_text = self.embedder(text)
        # Encode text, shape: (batch, max_seq_len, num_directions * hidden_size)
        encoded_text, (last_hidden, last_cell) = self.bilstm(
            embedded_text, sequence_length=seq_len)
        # Shape: (batch_size, lstm_hidden_size)
        hidden, att_weights = self.attention(encoded_text, mask)
        # Shape: (batch_size, fc_hidden_size)
        fc_out = paddle.tanh(self.fc(hidden))
        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc_out)
        return logits


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
        self.att_weight = self.create_parameter(
            shape=[1, hidden_size, 1], dtype='float32')

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
        att_weight = self.att_weight.tile(
            repeat_times=(paddle.shape(h)[0], 1, 1))
        # Shape: (batch_size, max_seq_len, 1)
        att_score = paddle.bmm(paddle.tanh(h), att_weight)
        if mask is not None:
            # mask, remove the effect of 'PAD'
            mask = paddle.cast(mask, dtype='float32')
            mask = mask.unsqueeze(axis=-1)
            inf_tensor = paddle.full(
                shape=mask.shape, dtype='float32', fill_value=-INF)
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
        self.bias = self.create_parameter(
            shape=[1, 1, hidden_size], dtype='float32')
        self.att_context_vector = self.create_parameter(
            shape=[1, hidden_size, 1], dtype='float32')

    def forward(self, input, mask=None):
        """
        Args:
            input (paddle.Tensor) of shape (batch, seq_len, input_size): Tensor containing the features of the input sequence.
            mask (paddle.Tensor) of shape (batch, seq_len) :
                Tensor is a bool tensor, whose each element identifies whether the input word id is pad token or not.
                Defaults to `None
        """
        weight = self.input_weight.tile(
            repeat_times=(paddle.shape(input)[0], 1, 1))
        bias = self.bias.tile(repeat_times=(paddle.shape(input)[0], 1, 1))
        # Shape: (batch_size, max_seq_len, hidden_size)
        word_squish = paddle.bmm(input, weight) + bias

        att_context_vector = self.att_context_vector.tile(
            repeat_times=(paddle.shape(input)[0], 1, 1))
        # Shape: (batch_size, max_seq_len, 1)
        att_score = paddle.bmm(word_squish, att_context_vector)
        if mask is not None:
            # mask, remove the effect of 'PAD'
            mask = paddle.cast(mask, dtype='float32')
            mask = mask.unsqueeze(axis=-1)
            inf_tensor = paddle.full(
                shape=paddle.shape(mask), dtype='float32', fill_value=-INF)
            att_score = paddle.multiply(att_score, mask) + paddle.multiply(
                inf_tensor, (1 - mask))
        att_weight = F.softmax(att_score, axis=1)

        # Shape: (batch_size, hidden_size)
        reps = paddle.bmm(input.transpose(perm=(0, 2, 1)),
                          att_weight).squeeze(-1)
        return reps, att_weight


class CNNModel(nn.Layer):
    """
    This class implements the Convolution Neural Network model.
    At a high level, the model starts by embedding the tokens and running them through
    a word embedding. Then, we encode these epresentations with a `CNNEncoder`.
    The CNN has one convolution layer for each ngram filter size. Each convolution operation gives
    out a vector of size num_filter. The number of times a convolution layer will be used
    is `num_tokens - ngram_size + 1`. The corresponding maxpooling layer aggregates all these
    outputs from the convolution layer and outputs the max. 
    Lastly, we take the output of the encoder to create a final representation,
    which is passed through some feed-forward layers to output a logits (`output_layer`).

    """

    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 num_filter=128,
                 ngram_filter_sizes=(3, ),
                 fc_hidden_size=96):
        super().__init__()
        self.embedder = nn.Embedding(
            vocab_size, emb_dim, padding_idx=padding_idx)
        self.encoder = nlp.seq2vec.CNNEncoder(
            emb_dim=emb_dim,
            num_filter=num_filter,
            ngram_filter_sizes=ngram_filter_sizes)
        self.fc = nn.Linear(self.encoder.get_output_dim(), fc_hidden_size)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, text, seq_len=None):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, len(ngram_filter_sizes)*num_filter)
        encoder_out = self.encoder(embedded_text)
        encoder_out = paddle.tanh(encoder_out)
        # Shape: (batch_size, fc_hidden_size)
        fc_out = self.fc(encoder_out)
        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc_out)
        return logits


class TextCNNModel(nn.Layer):
    """
    This class implements the Text Convolution Neural Network model.
    At a high level, the model starts by embedding the tokens and running them through
    a word embedding. Then, we encode these epresentations with a `CNNEncoder`.
    The CNN has one convolution layer for each ngram filter size. Each convolution operation gives
    out a vector of size num_filter. The number of times a convolution layer will be used
    is `num_tokens - ngram_size + 1`. The corresponding maxpooling layer aggregates all these
    outputs from the convolution layer and outputs the max. 
    Lastly, we take the output of the encoder to create a final representation,
    which is passed through some feed-forward layers to output a logits (`output_layer`).

    """

    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 num_filter=128,
                 ngram_filter_sizes=(1, 2, 3),
                 fc_hidden_size=96):
        super().__init__()
        self.embedder = nn.Embedding(
            vocab_size, emb_dim, padding_idx=padding_idx)
        self.encoder = nlp.seq2vec.CNNEncoder(
            emb_dim=emb_dim,
            num_filter=num_filter,
            ngram_filter_sizes=ngram_filter_sizes)
        self.fc = nn.Linear(self.encoder.get_output_dim(), fc_hidden_size)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, text, seq_len=None):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, len(ngram_filter_sizes)*num_filter)
        encoder_out = self.encoder(embedded_text)
        encoder_out = paddle.tanh(encoder_out)
        # Shape: (batch_size, fc_hidden_size)
        fc_out = paddle.tanh(self.fc(encoder_out))
        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc_out)
        return logits
