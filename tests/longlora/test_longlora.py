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
    set_group_size,
    ssa_scaled_dot_product_attention,
)

from .testing_utils import LLMTest


@parameterized_class(
    ["model_dir"],
    [
        ["llama"],  # 可以根据需要添加更多的模型目录
    ],
)
class TestSSA(LLMTest, unittest.TestCase):
    config_path: str = "./tests/fixtures/llm/predictor.yaml"
    model_dir: str = None

    def setUp(self) -> None:
        LLMTest.setUp(self)
        sys.path.insert(0, self.model_dir)

        # 设置 group size ratio
        self.ssa_group_size_ratio = 1 / 4
        set_group_size(self.ssa_group_size_ratio)

        # 创建输入张量的配置
        self.bsz = 2
        self.q_len = 16
        self.num_heads = 8
        self.head_dim = 64

        # 模拟查询、键、值状态
        self.query_states = paddle.randn([self.bsz, self.q_len, self.num_heads, self.head_dim])
        self.key_states = paddle.randn([self.bsz, self.q_len, self.num_heads, self.head_dim])
        self.value_states = paddle.randn([self.bsz, self.q_len, self.num_heads, self.head_dim])
        self.attention_mask = None

        self.config = type("Config", (object,), {"context_parallel_degree": 1})()

    def tearDown(self) -> None:
        LLMTest.tearDown(self)

    def test_ssa_attention_output_shape(self):
        # 运行SSA注意力机制
        attn_output = ssa_scaled_dot_product_attention(
            self.query_states,
            self.config,
            self.key_states,
            self.value_states,
            self.attention_mask,
            output_attentions=False,
        )
        print(attn_output.shape)
        # 验证输出形状是否符合预期
        self.assertEqual(attn_output.shape, [self.bsz, self.q_len, self.num_heads * self.head_dim])

    def test_ssa_attention_values_reasonable(self):
        attn_output = ssa_scaled_dot_product_attention(
            self.query_states,
            self.config,
            self.key_states,
            self.value_states,
            self.attention_mask,
            output_attentions=False,
        )
        print(attn_output.shape)

        # 确保输出数值在合理范围内
        self.assertFalse(paddle.isnan(attn_output).any().item())  # 无NaN
        self.assertFalse(paddle.isinf(attn_output).any().item())  # 无无穷值
