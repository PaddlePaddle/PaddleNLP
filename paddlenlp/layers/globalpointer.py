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


class RotaryPositionEmbedding(nn.Layer):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (paddle.arange(0, dim, 2, dtype="float32") / dim))
        t = paddle.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = paddle.matmul(t.unsqueeze(1), inv_freq.unsqueeze(0))
        self.register_buffer("sin", freqs.sin(), persistable=False)
        self.register_buffer("cos", freqs.cos(), persistable=False)

    def forward(self, x, offset=0):
        seqlen = paddle.shape(x)[-2]
        sin, cos = (
            self.sin[offset : offset + seqlen, :],
            self.cos[offset : offset + seqlen, :],
        )
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return paddle.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1).flatten(-2, -1)


class GlobalPointer(nn.Layer):
    def __init__(self, hidden_size, num_labels, pointer_inter_size=64, RoPE=True, tril_mask=True, max_length=512):
        super().__init__()
        self.num_labels = num_labels
        self.pointer_inter_size = pointer_inter_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask
        self.dense1 = nn.Linear(hidden_size, pointer_inter_size * 2)
        self.dense2 = nn.Linear(pointer_inter_size * 2, num_labels * 2)
        if RoPE:
            self.rotary = RotaryPositionEmbedding(pointer_inter_size, max_length)

    def forward(self, inputs, attention_mask=None):
        inputs = self.dense1(inputs)
        qw, kw = inputs[..., ::2], inputs[..., 1::2]
        if self.RoPE:
            qw, kw = self.rotary(qw), self.rotary(kw)

        logits = paddle.einsum("bmd,bnd->bmn", qw, kw) / self.pointer_inter_size**0.5
        bias = paddle.transpose(self.dense2(inputs), [0, 2, 1]) / 2
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]

        attn_mask = 1 - attention_mask[:, None, None, :] * attention_mask[:, None, :, None]
        logits = logits - attn_mask * 1e12

        if self.tril_mask:
            mask = paddle.tril(paddle.ones_like(logits), diagonal=-1)

            logits = logits - mask * 1e12

        return logits


class GlobalPointerForEntityExtraction(nn.Layer):
    def __init__(self, encoder, label_maps, pointer_inter_size=64):
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config["hidden_size"]
        gpcls = GlobalPointer
        self.entity_output = gpcls(hidden_size, len(label_maps["entity2id"]), pointer_inter_size=pointer_inter_size)

    def forward(self, input_ids, attention_mask):
        # input_ids, attention_mask, token_type_ids: (batch_size, seq_len)
        context_outputs = self.encoder(input_ids, attention_mask=attention_mask)
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]

        entity_output = self.entity_output(last_hidden_state, attention_mask)
        return [entity_output]


class GPLinkerForRelationExtraction(nn.Layer):
    def __init__(self, encoder, label_maps, pointer_inter_size=64):
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config["hidden_size"]
        num_ents = len(label_maps["entity2id"])
        if "relation2id" in label_maps.keys():
            num_rels = len(label_maps["relation2id"])
        else:
            num_rels = len(label_maps["sentiment2id"])
        gpcls = GlobalPointer

        self.entity_output = gpcls(hidden_size, num_ents, pointer_inter_size=pointer_inter_size)
        self.head_output = gpcls(
            hidden_size, num_rels, pointer_inter_size=pointer_inter_size, RoPE=False, tril_mask=False
        )
        self.tail_output = gpcls(
            hidden_size, num_rels, pointer_inter_size=pointer_inter_size, RoPE=False, tril_mask=False
        )

    def forward(self, input_ids, attention_mask):
        # input_ids, attention_mask, token_type_ids: (batch_size, seq_len)
        context_outputs = self.encoder(input_ids, attention_mask=attention_mask)
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]

        entity_output = self.entity_output(last_hidden_state, attention_mask)
        head_output = self.head_output(last_hidden_state, attention_mask)
        tail_output = self.tail_output(last_hidden_state, attention_mask)
        spo_output = [entity_output, head_output, tail_output]
        return spo_output
