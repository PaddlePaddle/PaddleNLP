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

import unittest

import numpy as np
from paddlenlp_ops import (
    fused_expert_moe,
    moe_expert_dispatch,
    moe_expert_ffn,
    moe_expert_reduce,
)

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.incubate.nn.functional import swiglu

paddle.seed(42)
np.random.seed(42)

class Expert(nn.Layer):
    def __init__(self, d_model, d_feedforward):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_feedforward * 2)
        self.fc2 = nn.Linear(d_feedforward, d_model)

    def forward(self, x, idx):
        x = self.fc1(x)
        x = swiglu(x)
        return self.fc2(x)

class TestSimpleFusedMoe(unittest.TestCase):
    def setUp(self):
        self.set_config()
        self.init_parameters()
        self.prepare_data()


    def set_config(self):
        self.dtype = "bfloat16"
        self.batch_size = 16
        self.seq_len = 128
        self.num_experts = 4
        self.d_model = 64
        self.d_feedforward = 128
        self.top_k = 2
        self.rtol = 1e-2
        self.atol = 1e-2

        paddle.set_default_dtype(self.dtype)

    def init_parameters(self):
        # 创建专家层
        self.experts = nn.LayerList([
            Expert(self.d_model, self.d_feedforward)
            for _ in range(self.num_experts)
        ])

        # 初始化门控权重
        self.gate = nn.Linear(self.d_model, self.num_experts)
        self.gate_weight = self.gate.weight.cast("float32")

    def prepare_data(self):
        """准备输入数据"""
        self.x = paddle.randn(
            [self.batch_size, self.seq_len, self.d_model],
            dtype=self.dtype
        )

        self.w0 = paddle.stack(
            [e.fc1.weight for e in self.experts], axis=0
        ).astype(self.dtype)
        self.b0 = paddle.stack(
            [e.fc1.bias for e in self.experts], axis=0
        ).reshape([self.num_experts, 1, -1]).astype(self.dtype)

        self.w1 = paddle.stack(
            [e.fc2.weight for e in self.experts], axis=0
        ).astype(self.dtype)
        self.b1 = paddle.stack(
            [e.fc2.bias for e in self.experts], axis=0
        ).reshape([self.num_experts, 1, -1]).astype(self.dtype)

    def baseline_forward(self, hidden_states):
        """（逐个专家计算）"""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape([-1, hidden_dim])

        # 路由计算
        logits = paddle.matmul(hidden_states.cast("float32"), self.gate_weight)
        weights = F.softmax(logits, axis=-1)
        routing_weights, selected_experts = paddle.topk(weights, self.top_k, axis=-1)
        # 结果累加
        final_hidden_states = paddle.zeros_like(hidden_states)

        expert_mask = paddle.transpose(
            F.one_hot(selected_experts, num_classes=self.num_experts), [2, 1, 0]
        )

        for expert_id in range(self.num_experts):
            expert_layer = self.experts[expert_id]
            idx, top_x = paddle.where(expert_mask[expert_id])

            current_state = paddle.index_select(
                hidden_states, top_x, axis=0
            ).reshape([-1, hidden_dim])
            current_hidden_states = (
                expert_layer(current_state, expert_id)
                * routing_weights[top_x, idx]
            )
            paddle.index_add_(
                x=final_hidden_states,
                index=top_x.squeeze(),
                axis=0,
                value=current_hidden_states.to(hidden_states.dtype),
            )
        final_hidden_states = paddle.reshape(
            final_hidden_states, [batch_size, seq_len, hidden_dim]
        )
        return final_hidden_states

    def fused_forward(self, x):
        """测试融合实现"""
        return fused_expert_moe(
            x,
            self.gate_weight,
            self.w0,
            self.w1,
            self.b0,
            None,
            self.b1,
            None,
            "None",
            self.top_k,
            False,
            False
        )

    def split_forward(self, hidden_states):
        '''测试拆分实现'''
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape([-1, hidden_dim])

        # 路由计算
        logits = paddle.matmul(hidden_states.cast("float32"), self.gate_weight)
        scores = F.softmax(logits, axis=-1)
        (
            permute_input,
            token_nums_per_expert,
            permute_indices_per_token,
            top_k_weights,
            top_k_indices,
        ) = moe_expert_dispatch(hidden_states, scores, self.top_k, False, topk_only_mode=True)

        ffn_out = moe_expert_ffn(
            permute_input,
            token_nums_per_expert,
            self.w0,
            self.w1,
            self.b0,
            None,
            None,
            "none",
        )
        output = moe_expert_reduce(
                ffn_out,
                top_k_weights,
                permute_indices_per_token,
                top_k_indices,
                None,
                norm_topk_prob=False,
                routed_scaling_factor=1.0,
            )
        output = paddle.reshape(output, [batch_size, seq_len, hidden_dim])
        return output

    def test_consistency(self):
        """测试一致性"""
        base_out = self.baseline_forward(self.x)
        fused_out = self.fused_forward(self.x)
        split_out = self.split_forward(self.x)

        np.testing.assert_allclose(
            base_out.cast("float32").numpy().astype("float32"),
            fused_out.cast("float32").numpy().astype("float32"),
            rtol=self.rtol,
            atol=self.atol
        )
        np.testing.assert_allclose(
            base_out.cast("float32").numpy().astype("float32"),
            split_out.cast("float32").numpy().astype("float32"),
            rtol=self.rtol,
            atol=self.atol
        )
        print("Test passed! Outputs match within tolerance.")

if __name__ == "__main__":
    unittest.main()
