#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class PinyinEmbedding(nn.Layer):
    def __init__(self, pinyin_map_len: int, embedding_size: int, pinyin_out_dim: int):
        """
            Pinyin Embedding Layer
        Args:
            embedding_size: the size of each embedding vector
            pinyin_out_dim: kernel number of conv
        """
        super(PinyinEmbedding, self).__init__()

        self.pinyin_out_dim = pinyin_out_dim
        self.embedding = nn.Embedding(pinyin_map_len, embedding_size)
        self.conv = nn.Conv1D(
            in_channels=embedding_size,
            out_channels=self.pinyin_out_dim,
            kernel_size=2,
            stride=1,
            padding=0,
        )

    def forward(self, pinyin_ids):
        """
        Args:
            pinyin_ids: (bs*sentence_length*pinyin_locs)

        Returns:
            pinyin_embed: (bs,sentence_length,pinyin_out_dim)
        """
        # input pinyin ids for 1-D conv
        embed = self.embedding(
            pinyin_ids
        )  # [bs,sentence_length*pinyin_locs,embed_size]
        bs, sentence_length, pinyin_locs, embed_size = embed.shape
        view_embed = embed.reshape(
            shape=[-1, pinyin_locs, embed_size]
        )  # [(bs*sentence_length),pinyin_locs,embed_size]
        input_embed = view_embed.transpose(
            [0, 2, 1]
        )  # [(bs*sentence_length), embed_size, pinyin_locs]
        # conv + max_pooling
        pinyin_conv = self.conv(input_embed)  # [(bs*sentence_length),pinyin_out_dim,H]
        pinyin_embed = F.max_pool1d(
            pinyin_conv, pinyin_conv.shape[-1]
        )  # [(bs*sentence_length),pinyin_out_dim,1]
        return pinyin_embed.reshape(
            shape=[bs, sentence_length, self.pinyin_out_dim]
        )  # [bs,sentence_length,pinyin_out_dim]


class GlyphEmbedding(nn.Layer):
    """Glyph2Image Embedding"""

    def __init__(self, num_embeddings, embedding_dim):
        super(GlyphEmbedding, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )

    def forward(self, input_ids):
        """
            get glyph images for batch inputs
        Args:
            input_ids: [batch, sentence_length]
        Returns:
            images: [batch, sentence_length, self.font_num*self.font_size*self.font_size]
        """
        # return self.embedding(input_ids).reshape([-1, self.font_num, self.font_size, self.font_size])
        return self.embedding(input_ids)


class FusionBertEmbeddings(nn.Layer):
    """
    Construct the embeddings from word, position, glyph, pinyin and token_type embeddings.
    """

    def __init__(
        self,
        vocab_size,
        hidden_size,
        pad_token_id,
        type_vocab_size,
        max_position_embeddings,
        pinyin_map_len,
        glyph_embedding_dim,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.1,
    ):
        super(FusionBertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_token_id
        )
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.pinyin_embeddings = PinyinEmbedding(
            pinyin_map_len=pinyin_map_len,
            embedding_size=128,
            pinyin_out_dim=hidden_size,
        )
        self.glyph_embeddings = GlyphEmbedding(vocab_size, glyph_embedding_dim)

        # self.LayerNorm is not snake-cased to stick with TensorFlow models variable name and be able to load
        # any TensorFlow checkpoint file
        self.glyph_map = nn.Linear(glyph_embedding_dim, hidden_size)
        self.map_fc = nn.Linear(hidden_size * 3, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids",
            paddle.arange(max_position_embeddings, dtype="int64").reshape(
                shape=[1, -1]
            ),
        )

    def forward(
        self, input_ids=None, pinyin_ids=None, token_type_ids=None, position_ids=None
    ):
        input_shape = input_ids.shape
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype="int64")
        # get char embedding, pinyin embedding and glyph embedding
        word_embeddings = self.word_embeddings(input_ids)  # [bs,l,hidden_size]

        pinyin_embeddings = self.pinyin_embeddings(
            pinyin_ids.reshape(shape=[input_shape[0], seq_length, 8])
        )  # [bs,l,hidden_size]

        glyph_embeddings = self.glyph_map(
            self.glyph_embeddings(input_ids)
        )  # [bs,l,hidden_size]
        # fusion layer
        concat_embeddings = paddle.concat(
            (word_embeddings, pinyin_embeddings, glyph_embeddings), axis=2
        )
        inputs_embeds = self.map_fc(concat_embeddings)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
