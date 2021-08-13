# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from utils import pad_sequence_paddle, mask_fill, index_sample
from model.dropouts import SharedDropout, IndependentDropout


class ErnieEncoder(nn.Layer):
    def __init__(self, 
                 args, 
                 pretrained_model):
        super(ErnieEncoder, self).__init__()
        self.ptm = pretrained_model
        self.args = args
        self.mlp_input_size = self.ptm.config["hidden_size"]

    def _flat_words(self, words):
        mask = words != self.args.pad_index
        lens = paddle.sum(paddle.cast(mask, "int32"), axis=-1)
        position = paddle.cumsum(lens + paddle.cast((lens == 0), "int32"), axis=1) - 1
        select = paddle.nonzero(mask)
        flat_words = paddle.gather_nd(words, select)
        flat_words = pad_sequence_paddle(
            paddle.split(flat_words,
                            paddle.sum(lens, axis=-1).numpy().tolist(), self.args.pad_index))
        max_len = flat_words.shape[1]
        position = mask_fill(position, position >= max_len, max_len - 1)
        return flat_words, position
    
    def forward(self, words, feat=None):
        words, position = self._flat_words(words)
        x, _ = self.ptm(words)

        x = paddle.reshape(
            index_sample(x, position),
            shape=position.shape[:2] + [x.shape[2]],
        )
        words = index_sample(words, position)
        return words, x


class LSTMByWPEncoder(nn.Layer):
    def __init__(self,
                 args,
                 lstm_by_wp_embed_size=200,
                 n_embed=300,
                 n_lstm_hidden=300,
                 n_lstm_layers=3,
                 lstm_dropout=0.33):
        super(LSTMByWPEncoder, self).__init__()
        self.args = args

        self.word_embed = nn.Embedding(self.args.n_words, lstm_by_wp_embed_size)

        self.lstm = nn.LSTM(
            input_size=lstm_by_wp_embed_size,
            hidden_size=n_lstm_hidden,
            num_layers=n_lstm_layers,
            dropout=lstm_dropout,
            direction="bidirectional"              
        )

        self.lstm_dropout = SharedDropout(p=lstm_dropout)
        self.mlp_input_size = n_lstm_hidden * 2

    def _flat_words(self, words):
        mask = words != self.args.pad_index
        lens = paddle.sum(paddle.cast(mask, "int32"), axis=-1)
        position = paddle.cumsum(lens + paddle.cast((lens == 0), "int32"), axis=1) - 1
        select = paddle.nonzero(mask)
        flat_words = paddle.gather_nd(words, select)
        flat_words = pad_sequence_paddle(
            paddle.split(flat_words,
                         paddle.sum(lens, axis=-1).numpy().tolist(), self.args.pad_index))
        max_len = flat_words.shape[1]
        position = mask_fill(position, position >= max_len, max_len - 1)
        return flat_words, position

    def forward(self, words, feats):
        words, position = self._flat_words(words)
        word_embed = self.word_embed(words)
        mask = words != self.args.pad_index
        seq_lens = paddle.sum(paddle.cast(mask, "int32"), axis=-1)

        x, _ = self.lstm(word_embed, sequence_length=seq_lens)
        x = paddle.reshape(
            index_sample(x, position),
            shape=position.shape[:2] + [x.shape[2]],
        )
        words = paddle.index_sample(words, position)
        x = self.lstm_dropout(x)
        return words, x


class LSTMEncoder(nn.Layer):
    def __init__(self, 
                 args,
                 n_char_embed=50,
                 n_feat_embed=60,
                 n_lstm_char_embed=100,
                 n_embed=300,
                 embed_dropout=0.33,
                 n_lstm_hidden=300,
                 n_lstm_layers=3,
                 lstm_dropout=0.33):
        super(LSTMEncoder, self).__init__()
        self.args = args

        if self.args.feat == "char":
            self.feat_embed = CharLSTMEncoder(
                n_chars=self.args.n_feats,
                n_embed=n_char_embed,
                n_out=n_lstm_char_embed,
                pad_index=self.args.feat_pad_index,
            )
            feat_embed_size = n_lstm_char_embed
        else:
            self.feat_embed = nn.Embedding(
                num_embeddings=self.args.n_feats, 
                embedding_dim=n_feat_embed)
            feat_embed_size = n_feat_embed          

        self.word_embed = nn.Embedding(
            num_embeddings=self.args.n_words, 
            embedding_dim=n_embed)
        self.embed_dropout = IndependentDropout(p=embed_dropout)

        self.lstm = nn.LSTM(
            input_size=n_embed + feat_embed_size,
            hidden_size=n_lstm_hidden,
            num_layers=n_lstm_layers,
            dropout=lstm_dropout,
            direction="bidirectional"              
        )
        self.lstm_dropout = SharedDropout(p=lstm_dropout)
        self.mlp_input_size = n_lstm_hidden * 2

    def forward(self, words, feats):
        word_embed = self.word_embed(words)
        feat_embed = self.feat_embed(feats)
        word_embed, feat_embed = self.embed_dropout(word_embed, feat_embed)
        embed = paddle.concat([word_embed, feat_embed], axis=-1)
        mask = words != self.args.pad_index
        seq_lens = paddle.sum(paddle.cast(mask, 'int32'))
        x, _ = self.lstm(embed, sequence_length=seq_lens)
        x = self.lstm_dropout(x)
        return words, x


class CharLSTMEncoder(nn.Layer):
    def __init__(self, 
                 n_chars, 
                 n_embed, 
                 n_out, 
                 pad_index=0):
        super(CharLSTMEncoder, self).__init__()
        self.n_chars = n_chars
        self.n_embed = n_embed
        self.n_out = n_out
        self.pad_index = pad_index

        # the embedding layer
        self.embed = nn.Embedding(num_embeddings=n_chars, embedding_dim=n_embed)
        # the lstm layer
        self.lstm = nn.LSTM(input_size=n_embed, hidden_size=n_out // 2, direction="bidirectional")

    def forward(self, x):
        """Forward network"""
        mask = paddle.any(x != self.pad_index, axis=-1)
        lens = paddle.sum(paddle.cast(mask, 'int32'))
        select = paddle.nonzero(mask)
        masked_x = paddle.gather_nd(x, select)
        char_mask = masked_x != self.pad_index
        emb = self.embed(masked_x)
        word_lens = paddle.sum(paddle.cast(char_mask, 'int32'))
        _, (h, _) = self.lstm(emb, sequence_length=word_lens)
        h = paddle.concat(paddle.unstack(h), axis=-1)
        feat_embed = pad_sequence_paddle(paddle.split(h, lens.numpy().tolist(), axis=0), self.pad_index)
        return feat_embed
        