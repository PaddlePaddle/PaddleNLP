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
import paddle.nn.functional as F

from ..attention_utils import _convert_param_attr_to_list
from .. import PretrainedModel, register_base_model

__all__ = [
    'ErnieDocModel',
    'ErnieDocPretrainedModel',
    'ErnieDocForSequenceClassification',
    'ErnieDocForTokenClassification',
    'ErnieDocForQuestionAnswering',
]


class PointwiseFFN(nn.Layer):
    def __init__(self,
                 d_inner_hid,
                 d_hid,
                 dropout_rate,
                 hidden_act,
                 weight_attr=None,
                 bias_attr=None):
        self.linear1 = nn.Linear(
            d_hid, d_inner_hid, weight_attr, bias_attr=bias_attr)
        self.dropout = nn.Dropout(dropout_rate, mode="upscale_in_train")
        self.linear2 = nn.Linear(
            d_inner_hid, d_hid, weight_attr, bias_attr=bias_attr)
        self.activation = getattr(F, hidden_act)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class MultiHeadAttention(nn.Layer):
    def __init__(self,
                 d_key,
                 d_value,
                 d_model,
                 n_head=1,
                 r_w_bias=None,
                 r_r_bias=None,
                 r_t_bias=None,
                 dropout_rate=0.,
                 weight_attr=None,
                 bias_attr=None):
        super(MultiHeadAttention, self).__init__()
        self.d_key = d_key
        self.d_value = d_value
        self.d_model = d_model
        self.n_head = n_head

        assert d_key * n_head == d_model, "d_model must be divisible by n_head"

        self.q_proj = nn.Linear(
            d_model,
            d_key * n_head,
            weight_attr=weight_attr,
            bias_attr=bias_attr)
        self.k_proj = nn.Linear(
            d_model,
            d_key * n_head,
            weight_attr=weight_attr,
            bias_attr=bias_attr)
        self.v_proj = nn.Linear(
            d_model,
            d_value * n_head,
            weight_attr=weight_attr,
            bias_attr=bias_attr)
        self.r_proj = nn.Linear(
            d_model,
            d_key * n_head,
            weight_attr=weight_attr,
            bias_attr=bias_attr)
        self.t_proj = nn.Linear(
            d_model,
            d_key * n_head,
            weight_attr=weight_attr,
            bias_attr=bias_attr)
        self.out_proj = nn.Linear(
            d_model, d_model, weight_attr=weight_attr, bias_attr=bias_attr)
        self.r_w_bias = r_w_bias.unsqueeze([0, 1])
        self.r_r_bias = r_r_bias.unsqueeze([0, 1])
        self.r_t_bias = r_t_bias.unsqueeze([0, 1])
        self.dropout = nn.Dropout(
            dropout_rate, mode="upscale_in_train") if dropout_rate else None

    def __compute_qkv(self, queries, keys, values, rel_pos, rel_task):
        q = self.q_proj(queries)
        k = self.k_proj(keys)
        v = self.v_proj(values)
        r = self.r_proj(rel_pos)
        t = self.t_proj(rel_task)
        return q, k, v, r, t

    def __split_heads(self, x, d_model, n_head):
        # x shape: [B, T, H]
        reshaped_x = paddle.reshape(x, shape=[0, 0, n_head, d_model // n_head])
        # shape: [B, N, T, HH]
        return paddle.transpose(x=reshaped_x, perm=[0, 2, 1, 3])

    def __rel_shift(self, x, klen=-1):
        # shape: [B, N, T, 2 * T + M]
        x_shape = x.shape
        x = paddle.reshape(x, [x_shape[0], x_shape[1], x_shape[3], x_shape[2]])
        x = x[:, :, 1:, :]
        x = paddle.reshape(
            x, [x_shape[0], x_shape[1], x_shape[2], x_shape[3] - 1])
        return x[:, :, :, :klen]

    def __scaled_dot_product_attention(self, q, k, v, r, t, attn_mask):
        q_w, q_r, q_t = q
        score_w = paddle.matmul(q_w, k, transpose_y=True)
        score_r = paddle.matmul(q_r, r, transpose_y=True)
        score_r = self.__rel_shift(score_r, k.shape[2])
        score_t = paddle.matmul(q_t, t, transpose_y=True)
        score = score_w + score_r + score_t
        score = score * (self.d_key**-0.5)
        if attn_mask is not None:
            score += attn_mask * -1e6
        weights = F.softmax(score)
        if self.dropout:
            weights = self.dropout(weights)
        out = paddle.matmul(weights, v)
        return out

    def __combine_heads(self, x):
        if len(x.shape) == 3: return x
        if len(x.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")
        # x shape: [B, N, T, HH]
        x = paddle.transpose(x, [0, 2, 1, 3])
        # target shape:[B, T, H]
        return paddle.reshape(x, [0, 0, x.shape[2] * x.shape[3]])

    def forward(self, queries, keys, values, rel_pos, rel_task, memory,
                attn_mask):
        if memory is not None and len(memory.shape) > 1:
            cat = paddle.concat([memorym, queries], 1)
        else:
            cat = queries
        keys, values = cat, cat

        if not (len(queries.shape) == len(keys.shape) == len(values.shape) \
            == len(rel_pos.shape) == len(rel_task.shape)== 3):
            raise ValueError(
                "Inputs: quries, keys, values, rel_pos and rel_task should all be 3-D tensors."
            )

        q, k, v, r, t = self.__compute_qkv(queries, keys, values, rel_pos,
                                           rel_task)
        q_w, q_r, q_t = list(
            map(lambda x: q + x, [self.r_w_bias, self.r_r_bias, self.r_t_bias]))
        q_w, q_r, q_t = list(
            map(lambda x: self.__split_heads(x, self.d_model, self.n_head),
                [q_w, q_r, q_t]))
        k, v, r, t = list(
            map(lambda x: self.__split_heads(x, self.d_model, self.n_head),
                [k, v, r, t]))

        ctx_multiheads = self.__scaled_dot_product_attention([q_w, q_r, q_t], \
                                    k, v, r, t, attn_mask)

        out = self.__combine_heads(ctx_multiheads)
        out = self.out_proj(out)
        return out


class ErnieDocEncoderLayer(nn.Layer):
    def __init__(self,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 hidden_act,
                 normalize_before=False,
                 epsilon=1e-5,
                 rel_pos_params_sharing=False,
                 r_w_bias=None,
                 r_r_bias=None,
                 r_t_bias=None,
                 weight_attr=None,
                 bias_attr=None):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(ErnieDocEncoderLayer, self).__init__()
        if rel_pos_params_sharing:
            assert (r_w_bias and r_r_bias and r_t_bias) is not None, \
                    "the rel pos bias can not be None when sharing the relative position params"
            self.r_w_bias, self.r_r_bias, self.r_t_bias = \
                r_w_bias, r_r_bias, r_t_bias
        else:
            self.r_w_bias, self.r_r_bias, self.r_t_bias = \
                list(map(lambda x: self.create_parameter(
                    shape=[n_head * d_key], dtype="float32")))

        weight_attrs = _convert_param_attr_to_list(weight_attr, 2)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 2)
        self.attn = MultiHeadAttention(
            d_key,
            d_value,
            d_model,
            n_head,
            self.r_w_bias,
            self.r_r_bias,
            self.r_t_bias,
            attention_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0], )
        self.ffn = PointwiseFFN(
            d_inner_hid,
            d_model,
            relu_dropout,
            hidden_act,
            weight_attr=weight_attrs[1],
            bias_attr=bias_attrs[1])
        self.norm1 = nn.LayerNorm(d_model, epsilon=epsilon)
        self.norm2 = nn.LayerNorm(d_model, epsilon=epsilon)
        self.dropout1 = nn.Dropout(
            prepostprocess_dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(
            prepostprocess_dropout, mode="upscale_in_train")

    def forward(self, enc_input, memory, rel_pos, rel_task, attn_mask):
        residual = enc_input
        if self.normalize_before:
            enc_input = self.norm1(enc_input)
        attn_output = self.attn(enc_input, enc_input, enc_input, rel_pos,
                                rel_task, memory, attn_mask)
        attn_output = residual + self.dropout1(attn_output)
        if not self.normalize_before:
            attn_output = self.norm1(attn_output)
        residual = attn_output
        if self.normalize_before:
            attn_output = self.norm2(attn_output)
        ffn_output = self.ffn(attn_output)
        output = residual + self.dropout2(ffn_output)
        if not self.normalize_before:
            output = self.norm2(output)
        return output


class ErnieDocEncoder(nn.Layer):
    def __init__(self, num_layers, encoder_layer, mem_len):
        super(ErnieDocEncoder, self).__init__()
        self.layers = nn.LayerList([(
            encoder_layer
            if i == 0 else type(encoder_layer)(**encoder_layer._config))
                                    for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(
            self.layers[0].d_model, epsilon=self.layers[0].epsilon)
        self.normalize_before = self.layers[0].normalize_before
        self.mem_len = mem_len

    def _cache_mem(curr_out, prev_mem):
        if self.mem_len is None or self.mem_len == 0:
            return None
        if prev_mem is None:
            new_mem = curr[:, -self.mem_len:, :]
        else:
            new_mem = paddle.concat([prev_mem, curr_out],
                                    1)[:, -self.mem_len:, :]
        new_mem.stop_gradient = True
        return new_mem

    def forward(self, enc_input, memories, rel_pos, rel_task, attn_mask):
        # no need to normalize enc_input, cause it's already normalized outside.
        new_mem = []
        for i, encoder_layer in enumerate(self.layers):
            enc_input = encoder_layer(enc_input, memories[i], rel_pos, rel_task,
                                      attn_mask)
            new_mem += [self._cache_mem(enc_input, memories[i])]

        return enc_input, new_mem


class ErnieDocPretrainedModel(PretrainedModel):
    pass


@register_base_model
class ErnieDocModel(ErnieDocPretrainedModel):
    def __init__(self):
        pass

    def forward(self):
        pass


class ErnieDocForSequenceClassification(ErnieDocPretrainedModel):
    def __init__(self):
        pass

    def forward(self):
        pass


class ErnieDocForTokenClassification(ErnieDocPretrainedModel):
    def __init__(self):
        pass

    def forward(self):
        pass


class ErnieDocForQuestionAnswering(ErnieDocPretrainedModel):
    def __init__(self):
        pass

    def forward(self):
        pass
