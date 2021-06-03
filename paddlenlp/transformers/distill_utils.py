# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2020 Huawei Technologies Co., Ltd.
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

import math

from paddle import tensor
import paddle.nn.functional as F
from paddle.nn import MultiHeadAttention, TransformerEncoderLayer, TransformerEncoder
from paddle.fluid.data_feeder import convert_dtype
from paddlenlp.utils.log import logger

__all__ = ['to_distill']


def to_distill(self):
    """
    Can be bound to object with transformer layers, and exposes attributes
    `outputs.hidden_states`and `outputs.attentions` of the object for
    distillation.
    """
    logger.warning("to_distill is an experimental API and subject to change.")
    MultiHeadAttention._forward = attention_forward
    TransformerEncoderLayer._forward = transformer_encoder_layer_forward
    TransformerEncoder._forward = transformer_encoder_forward

    def init_forward(layer):
        if isinstance(layer, (MultiHeadAttention, TransformerEncoderLayer,
                              TransformerEncoder)):
            layer.forward = layer._forward

    for layer in self.children():
        layer.apply(init_forward)

    self.outputs = getattr(self, self.base_model_prefix).encoder
    return self


def _convert_attention_mask(attn_mask, dtype):
    if attn_mask is not None and attn_mask.dtype != dtype:
        attn_mask_dtype = convert_dtype(attn_mask.dtype)
        if attn_mask_dtype == 'bool' or 'int' in attn_mask_dtype:
            attn_mask = (paddle.cast(attn_mask, dtype) - 1.0) * 1e9
        else:
            attn_mask = paddle.cast(attn_mask, dtype)
    return attn_mask


def attention_forward(self,
                      query,
                      key=None,
                      value=None,
                      attn_mask=None,
                      cache=None):
    """
    Redefines the `forward` function of `paddle.nn.MultiHeadAttention`
    """
    key = query if key is None else key
    value = query if value is None else value
    # compute q ,k ,v
    if cache is None:
        q, k, v = self._prepare_qkv(query, key, value, cache)
    else:
        q, k, v, cache = self._prepare_qkv(query, key, value, cache)

    # scale dot product attention
    product = tensor.matmul(x=q, y=k, transpose_y=True)
    product /= math.sqrt(self.head_dim)

    if attn_mask is not None:
        # Support bool or int mask
        attn_mask = _convert_attention_mask(attn_mask, product.dtype)
        product = product + attn_mask

    self.attention_matrix = product
    weights = F.softmax(product)
    if self.dropout:
        weights = F.dropout(
            weights,
            self.dropout,
            training=self.training,
            mode="upscale_in_train")

    out = tensor.matmul(weights, v)

    # combine heads
    out = tensor.transpose(out, perm=[0, 2, 1, 3])
    out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

    # project to output
    out = self.out_proj(out)

    outs = [out]
    if self.need_weights:
        outs.append(weights)
    if cache is not None:
        outs.append(cache)
    return out if len(outs) == 1 else tuple(outs)


def transformer_encoder_layer_forward(self, src, src_mask=None, cache=None):
    """
    Redefines the `forward` function of `paddle.nn.TransformerEncoderLayer`
    """
    src_mask = _convert_attention_mask(src_mask, src.dtype)

    residual = src
    if self.normalize_before:
        src = self.norm1(src)
    # Add cache for encoder for the usage like UniLM
    if cache is None:
        src = self.self_attn(src, src, src, src_mask)
    else:
        src, incremental_cache = self.self_attn(src, src, src, src_mask, cache)
    self.attention_matrix = self.self_attn.attention_matrix
    src = residual + self.dropout1(src)
    if not self.normalize_before:
        src = self.norm1(src)

    residual = src
    if self.normalize_before:
        src = self.norm2(src)
    src = self.linear2(self.dropout(self.activation(self.linear1(src))))
    src = residual + self.dropout2(src)
    if not self.normalize_before:
        src = self.norm2(src)
    return src if cache is None else (src, incremental_cache)


def transformer_encoder_forward(self, src, src_mask=None, cache=None):
    """
    Redefines the `forward` function of `paddle.nn.TransformerEncoder`
    """
    src_mask = _convert_attention_mask(src_mask, src.dtype)

    output = src
    new_caches = []
    self.attentions = []
    self.hidden_states = []
    self.v = None
    for i, mod in enumerate(self.layers):
        self.hidden_states.append(output)
        if cache is None:
            output = mod(output, src_mask=src_mask)
        else:
            output, new_cache = mod(output, src_mask=src_mask, cache=cache[i])
            new_caches.append(new_cache)
        self.attentions.append(mod.attention_matrix)

    if self.norm is not None:
        output = self.norm(output)
    self.hidden_states.append(output)

    return output if cache is None else (output, new_caches)
