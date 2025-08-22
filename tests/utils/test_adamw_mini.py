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
import paddle

from paddlenlp.utils.optimizer import AdamWMini


class SimpleTransformerPaddle(paddle.nn.Layer):
    def __init__(self, dim=2048, n_heads=32, vocab_size=100, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # Embedding layer
        self.embd = paddle.nn.Embedding(vocab_size, dim, weight_attr=paddle.nn.initializer.Normal(mean=0.0, std=0.02))

        # Query/Key/Value projections
        self.wq = paddle.nn.Linear(dim, dim, weight_attr=paddle.nn.initializer.Normal(mean=0.0, std=0.02))
        self.wk = paddle.nn.Linear(dim, dim, weight_attr=paddle.nn.initializer.Normal(mean=0.0, std=0.02))
        self.wv = paddle.nn.Linear(dim, dim, weight_attr=paddle.nn.initializer.Normal(mean=0.0, std=0.02))

        # Attention projection
        self.wo = paddle.nn.Linear(dim, dim, weight_attr=paddle.nn.initializer.Normal(mean=0.0, std=0.02))

        # LayerNorm layers
        self.ln1 = paddle.nn.LayerNorm(dim)
        self.ln2 = paddle.nn.LayerNorm(dim)

        # MLP layers
        self.mlp = paddle.nn.Sequential(
            paddle.nn.Linear(dim, 4 * dim, weight_attr=paddle.nn.initializer.Normal(mean=0.0, std=0.02)),
            paddle.nn.ReLU(),
            paddle.nn.Linear(4 * dim, dim, weight_attr=paddle.nn.initializer.Normal(mean=0.0, std=0.02)),
        )

        # Output layer
        self.lm_head = paddle.nn.Linear(dim, vocab_size, weight_attr=paddle.nn.initializer.Normal(mean=0.0, std=0.02))

        # Bias parameters
        self.bias = paddle.create_parameter(
            [dim], dtype=dtype, default_initializer=paddle.nn.initializer.Constant(value=0.0)
        )

    def forward(self, input_ids):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # Embedding
        hidden_states = self.embd(input_ids)  # [batch_size, seq_len, dim]

        # Query/Key/Value projections and reshape for multi-head attention
        query = self.wq(hidden_states)  # [batch_size, seq_len, dim]
        key = self.wk(hidden_states)  # [batch_size, seq_len, dim]
        value = self.wv(hidden_states)  # [batch_size, seq_len, dim]

        # Reshape to [batch_size, seq_len, n_heads, head_dim]
        query = query.reshape([batch_size, seq_len, self.n_heads, self.head_dim])
        key = key.reshape([batch_size, seq_len, self.n_heads, self.head_dim])
        value = value.reshape([batch_size, seq_len, self.n_heads, self.head_dim])

        # Transpose to [batch_size, n_heads, seq_len, head_dim]
        query = query.transpose([0, 2, 1, 3])
        key = key.transpose([0, 2, 1, 3])
        value = value.transpose([0, 2, 1, 3])

        # Scaled dot-product attention
        scale = self.head_dim**-0.5
        attn_weights = paddle.matmul(query * scale, key.transpose([0, 1, 3, 2]))
        attn_weights = paddle.nn.functional.softmax(attn_weights, axis=-1)

        # Apply attention to values
        attn_output = paddle.matmul(attn_weights, value)  # [batch_size, n_heads, seq_len, head_dim]

        # Reshape back to [batch_size, seq_len, dim]
        attn_output = attn_output.transpose([0, 2, 1, 3])
        attn_output = attn_output.reshape([batch_size, seq_len, self.dim])

        # Attention output projection with residual connection and layer norm
        attn_output = self.wo(attn_output)
        hidden_states = self.ln1(hidden_states + attn_output)

        # Feed forward with residual connection and layer norm
        feed_forward = self.mlp(hidden_states)
        hidden_states = self.ln2(hidden_states + feed_forward)

        # Output
        output = self.lm_head(hidden_states + self.bias)

        return output


def generate_data(batch_size=32, seq_len=64, vocab_size=100):
    x = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    y = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    return x, y


class TestAdamWMini(unittest.TestCase):
    def setUp(self):
        # Set random seeds for reproducibility
        SEED = 1
        np.random.seed(SEED)
        paddle.seed(SEED)
        DTYPE = "float32"
        paddle.set_default_dtype(DTYPE)

    def test_adamw_mini(self):
        lr = 1e-3
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        weight_decay = 0.01
        dim = 2048
        n_heads = 32
        model = SimpleTransformerPaddle()

        optimizer = AdamWMini(
            model.named_parameters(),
            learning_rate=lr,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            dim=dim,
            n_heads=n_heads,
        )

        for _ in range(2):
            x_np, _ = generate_data()
            x = paddle.to_tensor(x_np, dtype="int64")

            output = model(x)
            loss = paddle.mean(output)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
