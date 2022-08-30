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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import paddlenlp as nlp


class SimNet(nn.Layer):

    def __init__(self,
                 network,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 pad_token_id=0):
        super().__init__()

        network = network.lower()
        if network == 'bow':
            self.model = BoWModel(vocab_size,
                                  num_classes,
                                  emb_dim,
                                  padding_idx=pad_token_id)
        elif network == 'cnn':
            self.model = CNNModel(vocab_size,
                                  num_classes,
                                  emb_dim,
                                  padding_idx=pad_token_id)
        elif network == 'gru':
            self.model = GRUModel(vocab_size,
                                  num_classes,
                                  emb_dim,
                                  direction='forward',
                                  padding_idx=pad_token_id)
        elif network == 'lstm':
            self.model = LSTMModel(vocab_size,
                                   num_classes,
                                   emb_dim,
                                   direction='forward',
                                   padding_idx=pad_token_id)
        else:
            raise ValueError(
                "Unknown network: %s, it must be one of bow, cnn, lstm or gru."
                % network)

    def forward(self, query, title, query_seq_len=None, title_seq_len=None):
        logits = self.model(query, title, query_seq_len, title_seq_len)
        return logits

    def forward_interpret(self,
                          query,
                          title,
                          query_seq_len=None,
                          title_seq_len=None,
                          noise=None,
                          i=None,
                          n_samples=None):

        logits, addiational_info = self.model.forward_interpreter(
            query,
            title,
            query_seq_len,
            title_seq_len,
            noise=noise,
            i=i,
            n_samples=n_samples)

        return logits, addiational_info['attention'], addiational_info[
            'embedded']


class BoWModel(nn.Layer):
    """
    This class implements the Bag of Words Classification Network model to classify texts.
    At a high level, the model starts by embedding the tokens and running them through
    a word embedding. Then, we encode these epresentations with a `BoWEncoder`.
    Lastly, we take the output of the encoder to create a final representation,
    which is passed through some feed-forward layers to output a logits (`output_layer`).
    Args:
        vocab_size (obj:`int`): The vocabulary size.
        emb_dim (obj:`int`, optional, defaults to 128):  The embedding dimension.
        padding_idx (obj:`int`, optinal, defaults to 0) : The pad token index.
        hidden_size (obj:`int`, optional, defaults to 128): The first full-connected layer hidden size.
        fc_hidden_size (obj:`int`, optional, defaults to 96): The second full-connected layer hidden size.
        num_classes (obj:`int`): All the labels that the data has.
    """

    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 fc_hidden_size=128):
        super().__init__()
        self.embedder = nn.Embedding(vocab_size,
                                     emb_dim,
                                     padding_idx=padding_idx)
        self.bow_encoder = nlp.seq2vec.BoWEncoder(emb_dim)
        self.fc = nn.Linear(self.bow_encoder.get_output_dim() * 2,
                            fc_hidden_size)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, query, title, query_seq_len=None, title_seq_len=None):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_query = self.embedder(query)
        embedded_title = self.embedder(title)
        # Shape: (batch_size, embedding_dim)
        summed_query = self.bow_encoder(embedded_query)
        summed_title = self.bow_encoder(embedded_title)
        encoded_query = paddle.tanh(summed_query)
        encoded_title = paddle.tanh(summed_title)
        # Shape: (batch_size, embedding_dim*2)
        contacted = paddle.concat([encoded_query, encoded_title], axis=-1)
        # Shape: (batch_size, fc_hidden_size)
        fc_out = paddle.tanh(self.fc(contacted))
        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc_out)
        # probs = F.softmax(logits, axis=-1)
        return logits


class LSTMModel(nn.Layer):

    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 lstm_hidden_size=128,
                 direction='forward',
                 lstm_layers=1,
                 dropout_rate=0.0,
                 pooling_type=None,
                 fc_hidden_size=128):
        super().__init__()
        self.embedder = nn.Embedding(num_embeddings=vocab_size,
                                     embedding_dim=emb_dim,
                                     padding_idx=padding_idx)
        self.lstm_encoder = nlp.seq2vec.LSTMEncoder(emb_dim,
                                                    lstm_hidden_size,
                                                    num_layers=lstm_layers,
                                                    direction=direction,
                                                    dropout=dropout_rate)
        self.fc = nn.Linear(self.lstm_encoder.get_output_dim() * 2,
                            fc_hidden_size)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)
        self.pad_token_id = padding_idx

    def forward(self, query, title, query_seq_len, title_seq_len):
        assert query_seq_len is not None and title_seq_len is not None
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_query = self.embedder(query)
        embedded_title = self.embedder(title)
        # Shape: (batch_size, lstm_hidden_size)
        query_repr = self.lstm_encoder(embedded_query,
                                       sequence_length=query_seq_len)
        title_repr = self.lstm_encoder(embedded_title,
                                       sequence_length=title_seq_len)
        # Shape: (batch_size, 2*lstm_hidden_size)
        contacted = paddle.concat([query_repr, title_repr], axis=-1)
        # Shape: (batch_size, fc_hidden_size)
        fc_out = paddle.tanh(self.fc(contacted))
        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc_out)
        # probs = F.softmax(logits, axis=-1)

        return logits

    def forward_interpreter(self,
                            query,
                            title,
                            query_seq_len,
                            title_seq_len,
                            noise=None,
                            i=None,
                            n_samples=None):
        assert query_seq_len is not None and title_seq_len is not None
        # Shape: (batch_size, num_tokens, embedding_dim)

        query_baseline = paddle.to_tensor([self.pad_token_id] *
                                          query.shape[1]).unsqueeze(0)
        title_baseline = paddle.to_tensor([self.pad_token_id] *
                                          title.shape[1]).unsqueeze(0)

        embedded_query = self.embedder(query)
        embedded_title = self.embedder(title)
        embedded_query_baseline = self.embedder(query_baseline)
        embedded_title_baseline = self.embedder(title_baseline)

        if noise is not None and noise.upper() == 'INTEGRATED':
            embedded_query = embedded_query_baseline + i / (n_samples - 1) * (
                embedded_query - embedded_query_baseline)
            embedded_title = embedded_title_baseline + i / (n_samples - 1) * (
                embedded_title - embedded_title_baseline)

        # Shape: (batch_size, lstm_hidden_size)
        query_repr = self.lstm_encoder(embedded_query,
                                       sequence_length=query_seq_len)
        title_repr = self.lstm_encoder(embedded_title,
                                       sequence_length=title_seq_len)
        # Shape: (batch_size, 2*lstm_hidden_size)
        contacted = paddle.concat([query_repr, title_repr], axis=-1)
        # Shape: (batch_size, fc_hidden_size)
        fc_out = paddle.tanh(self.fc(contacted))
        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc_out)
        probs = F.softmax(logits, axis=-1)

        q_att = paddle.matmul(fc_out, embedded_query, transpose_y=True).squeeze(
            axis=[1])  # (bsz, query_len)
        q_att = F.softmax(q_att, axis=-1)
        t_att = paddle.matmul(fc_out, embedded_title, transpose_y=True).squeeze(
            axis=[1])  # (bsz, title_len)
        t_att = F.softmax(t_att, axis=-1)

        addiational_info = {
            'embedded': [embedded_query, embedded_title],
            'attention': [q_att, t_att],
        }
        # return logits, addiational_info
        return probs, addiational_info


class GRUModel(nn.Layer):

    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 gru_hidden_size=128,
                 direction='forward',
                 gru_layers=1,
                 dropout_rate=0.0,
                 pooling_type=None,
                 fc_hidden_size=96):
        super().__init__()
        self.embedder = nn.Embedding(num_embeddings=vocab_size,
                                     embedding_dim=emb_dim,
                                     padding_idx=padding_idx)
        self.gru_encoder = nlp.seq2vec.GRUEncoder(emb_dim,
                                                  gru_hidden_size,
                                                  num_layers=gru_layers,
                                                  direction=direction,
                                                  dropout=dropout_rate)
        self.fc = nn.Linear(self.gru_encoder.get_output_dim() * 2,
                            fc_hidden_size)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, query, title, query_seq_len, title_seq_len):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_query = self.embedder(query)
        embedded_title = self.embedder(title)
        # Shape: (batch_size, gru_hidden_size)
        query_repr = self.gru_encoder(embedded_query,
                                      sequence_length=query_seq_len)
        title_repr = self.gru_encoder(embedded_title,
                                      sequence_length=title_seq_len)
        # Shape: (batch_size, 2*gru_hidden_size)
        contacted = paddle.concat([query_repr, title_repr], axis=-1)
        # Shape: (batch_size, fc_hidden_size)
        fc_out = paddle.tanh(self.fc(contacted))
        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc_out)
        # probs = F.softmax(logits, axis=-1)

        return logits


class CNNModel(nn.Layer):
    """
    This class implements the


     Convolution Neural Network model.
    At a high level, the model starts by embedding the tokens and running them through
    a word embedding. Then, we encode these epresentations with a `CNNEncoder`.
    The CNN has one convolution layer for each ngram filter size. Each convolution operation gives
    out a vector of size num_filter. The number of times a convolution layer will be used
    is `num_tokens - ngram_size + 1`. The corresponding maxpooling layer aggregates all these
    outputs from the convolution layer and outputs the max. 
    Lastly, we take the output of the encoder to create a final representation,
    which is passed through some feed-forward layers to output a logits (`output_layer`).
    Args:
        vocab_size (obj:`int`): The vocabulary size.
        emb_dim (obj:`int`, optional, defaults to 128):  The embedding dimension.
        padding_idx (obj:`int`, optinal, defaults to 0) : The pad token index.
        num_classes (obj:`int`): All the labels that the data has.
    """

    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 num_filter=256,
                 ngram_filter_sizes=(3, ),
                 fc_hidden_size=128):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedder = nn.Embedding(vocab_size,
                                     emb_dim,
                                     padding_idx=padding_idx)
        self.encoder = nlp.seq2vec.CNNEncoder(
            emb_dim=emb_dim,
            num_filter=num_filter,
            ngram_filter_sizes=ngram_filter_sizes)
        self.fc = nn.Linear(self.encoder.get_output_dim() * 2, fc_hidden_size)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, query, title, query_seq_len=None, title_seq_len=None):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_query = self.embedder(query)
        embedded_title = self.embedder(title)
        # Shape: (batch_size, num_filter)
        query_repr = self.encoder(embedded_query)
        title_repr = self.encoder(embedded_title)
        # Shape: (batch_size, 2*num_filter)
        contacted = paddle.concat([query_repr, title_repr], axis=-1)
        # Shape: (batch_size, fc_hidden_size)
        fc_out = paddle.tanh(self.fc(contacted))
        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc_out)
        # probs = F.softmax(logits, axis=-1)
        return logits
