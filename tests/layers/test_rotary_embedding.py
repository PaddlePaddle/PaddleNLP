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

from paddlenlp.layers import RotaryEmbedding


class TestRotaryEmbedding(unittest.TestCase):
    batch_size, seq_length, num_heads, head_dim = 2, 7, 2, 4
    tensor_shape = [batch_size, seq_length, num_heads, head_dim]

    def test_forward(self):
        position_ids = paddle.arange(self.seq_length).unsqueeze(0)
        query_states = paddle.rand(self.tensor_shape)
        key_states = paddle.rand(self.tensor_shape)
        rope_embedding = RotaryEmbedding(dim=self.head_dim, max_position_embeddings=16)
        cos, sin = rope_embedding(key_states, seq_len=self.seq_length)
        query_states, key_states = RotaryEmbedding.apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        self.assertEqual(query_states.shape, self.tensor_shape)
        self.assertEqual(key_states.shape, self.tensor_shape)

    # NOTE: Running this test on GPU causes CUDA Error and fails all other tests...
    def test_exceed_max_length(self):
        position_ids = paddle.arange(self.seq_length).unsqueeze(0)
        query_states = paddle.rand(self.tensor_shape)
        key_states = paddle.rand(self.tensor_shape)
        # small max_position_embeddings
        rope_embedding = RotaryEmbedding(dim=self.head_dim, max_position_embeddings=3)
        cos, sin = rope_embedding(key_states, seq_len=self.seq_length)
        with self.assertRaises(ValueError):
            query_states, key_states = RotaryEmbedding.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )

    def test_2d_position_id(self):
        # 1D position_id
        position_ids = paddle.arange(self.seq_length)
        query_states = paddle.rand(self.tensor_shape)
        key_states = paddle.rand(self.tensor_shape)
        rope_embedding = RotaryEmbedding(dim=self.head_dim, max_position_embeddings=16)
        cos, sin = rope_embedding(key_states, seq_len=self.seq_length)
        with self.assertRaisesRegex(ValueError, "position_ids should be a 2D tensor"):
            query_states, key_states = RotaryEmbedding.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )
