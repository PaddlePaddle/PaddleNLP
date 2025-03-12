# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import sys
import unittest

import paddle
from parameterized import parameterized_class

from paddlenlp.transformers.longlora import (
    replace_llama_attn,
    ssa_scaled_dot_product_attention,
)

from .testing_utils import LLMTest


@parameterized_class(
    ["model_dir"],
    [
        ["llama"],
    ],
)
class TestSSA(LLMTest, unittest.TestCase):
    config_path: str = "./tests/fixtures/llm/predictor.yaml"
    model_dir: str = None

    def setUp(self) -> None:
        LLMTest.setUp(self)
        sys.path.insert(0, self.model_dir)

        self.ssa_group_size_ratio = 1 / 4
        replace_llama_attn(self.ssa_group_size_ratio, use_ssa=True)

        self.bsz = 2
        self.q_len = 16
        self.num_heads = 8
        self.head_dim = 64

        self.query_states = paddle.randn([self.bsz, self.q_len, self.num_heads, self.head_dim])
        self.key_states = paddle.randn([self.bsz, self.q_len, self.num_heads, self.head_dim])
        self.value_states = paddle.randn([self.bsz, self.q_len, self.num_heads, self.head_dim])
        self.attention_mask = None

        self.config = type("Config", (object,), {"context_parallel_degree": 1})()

    def tearDown(self) -> None:
        LLMTest.tearDown(self)

    def test_ssa_attention_output_shape(self):
        attn_output = ssa_scaled_dot_product_attention(
            self.query_states,
            self.config,
            self.key_states,
            self.value_states,
            self.attention_mask,
            output_attentions=False,
            ssa_group_size_ratio=self.ssa_group_size_ratio,
        )
        self.assertEqual(attn_output.shape, [self.bsz, self.q_len, self.num_heads * self.head_dim])

    def test_ssa_attention_values_reasonable(self):
        attn_output = ssa_scaled_dot_product_attention(
            self.query_states,
            self.config,
            self.key_states,
            self.value_states,
            self.attention_mask,
            output_attentions=False,
            ssa_group_size_ratio=self.ssa_group_size_ratio,
        )

        self.assertFalse(paddle.isnan(attn_output).any().item())  # 无NaN
        self.assertFalse(paddle.isinf(attn_output).any().item())  # 无无穷值
