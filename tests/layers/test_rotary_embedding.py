# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import paddle

from paddlenlp.layers import (
    DynamicNTKScalingRotaryEmbedding,
    LinearScalingRotaryEmbedding,
    NTKScalingRotaryEmbedding,
    RotaryEmbedding,
)


class RotaryEmbeddingsTestCommon:
    batch_size, seq_length, num_heads, head_dim = 2, 7, 2, 4
    tensor_shape = [batch_size, seq_length, num_heads, head_dim]
    query_states = paddle.rand(tensor_shape)
    key_states = paddle.rand(tensor_shape)
    classname = None

    def test_forward(self):
        max_position_embeddings = 16
        position_ids = paddle.arange(self.seq_length).unsqueeze(0)

        rope_embedding = self.classname(dim=self.head_dim, max_position_embeddings=max_position_embeddings)
        cos, sin = rope_embedding(self.key_states, seq_len=self.seq_length)
        query_states, key_states = self.classname.apply_rotary_pos_emb(
            self.query_states, self.key_states, cos, sin, position_ids
        )
        self.assertEqual(rope_embedding.max_seq_len_cached, max_position_embeddings)
        self.assertEqual(query_states.shape, self.tensor_shape)
        self.assertEqual(key_states.shape, self.tensor_shape)

    def test_exceed_max_length(self):
        # small max_position_embeddings
        max_position_embeddings = 3
        position_ids = paddle.arange(self.seq_length).unsqueeze(0)
        rope_embedding = self.classname(dim=self.head_dim, max_position_embeddings=max_position_embeddings)
        self.assertEqual(rope_embedding.max_seq_len_cached, max_position_embeddings)
        cos, sin = rope_embedding(self.key_states, seq_len=self.seq_length)
        query_states, key_states = self.classname.apply_rotary_pos_emb(
            self.query_states, self.key_states, cos, sin, position_ids
        )
        self.assertEqual(rope_embedding.max_seq_len_cached, self.seq_length)
        self.assertEqual(query_states.shape, self.tensor_shape)
        self.assertEqual(key_states.shape, self.tensor_shape)

    def test_2d_position_id(self):
        # 1D position_id
        position_ids = paddle.arange(self.seq_length)
        rope_embedding = self.classname(dim=self.head_dim, max_position_embeddings=16)
        cos, sin = rope_embedding(self.key_states, seq_len=self.seq_length)
        with self.assertRaisesRegex(ValueError, "position_ids should be a 2D tensor"):
            query_states, key_states = self.classname.apply_rotary_pos_emb(
                self.query_states, self.key_states, cos, sin, position_ids
            )


class TestRotaryRotaryEmbeddings(RotaryEmbeddingsTestCommon, unittest.TestCase):
    classname = RotaryEmbedding


class TestLinearScalingRotaryEmbedding(RotaryEmbeddingsTestCommon, unittest.TestCase):
    classname = LinearScalingRotaryEmbedding


class TestNTKScalingRotaryEmbedding(RotaryEmbeddingsTestCommon, unittest.TestCase):
    classname = NTKScalingRotaryEmbedding


class TestDynamicNTKScalingRotaryEmbedding(RotaryEmbeddingsTestCommon, unittest.TestCase):
    classname = DynamicNTKScalingRotaryEmbedding

    def test_exceed_max_length(self):
        # small max_position_embeddings
        max_position_embeddings = 3
        position_ids = paddle.arange(self.seq_length).unsqueeze(0)
        rope_embedding = self.classname(
            dim=self.head_dim, max_position_embeddings=max_position_embeddings, scaling_factor=2.0
        )
        cos, sin = rope_embedding(self.key_states, seq_len=self.seq_length)
        query_states, key_states = self.classname.apply_rotary_pos_emb(
            self.query_states, self.key_states, cos, sin, position_ids
        )
        self.assertEqual(rope_embedding.max_seq_len_cached, max_position_embeddings)
        self.assertEqual(query_states.shape, self.tensor_shape)
        self.assertEqual(key_states.shape, self.tensor_shape)
