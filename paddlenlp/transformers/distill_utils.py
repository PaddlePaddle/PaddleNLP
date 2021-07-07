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

import math

import paddle
from paddle import tensor
import paddle.nn.functional as F
from paddle.nn import MultiHeadAttention, TransformerEncoderLayer, TransformerEncoder
from paddle.fluid.data_feeder import convert_dtype
from paddlenlp.utils.log import logger
from paddlenlp.transformers import TinyBertForPretraining

__all__ = ['to_distill', 'calc_minilm_loss']


def calc_minilm_loss(loss_fct, s, t):
    head_num, pad_seq_len = s.shape[1], s.shape[2]
    s_head_dim, t_head_dim = s.shape[3], t.shape[3]
    scaled_dot_product_s = tensor.matmul(
        x=s, y=s, transpose_y=True) / math.sqrt(s_head_dim)
    del s
    scaled_dot_product_t = tensor.matmul(
        x=t, y=t, transpose_y=True) / math.sqrt(t_head_dim)
    del t
    loss = loss_fct(
        F.log_softmax(scaled_dot_product_s), F.softmax(scaled_dot_product_t))
    loss /= head_num * pad_seq_len
    return loss


def to_distill(self, num_relation_heads=0):
    """
    Can be bound to object with transformer encoder layers, and make model
    expose attributes `outputs.qs`, `outputs.ks`, `outputs.vs`,
    `outputs.scaled_qks`, `outputs.hidden_states`and `outputs.attentions` of
    the object for distillation.
    """
    logger.warning("`to_distill` is an experimental API and subject to change.")
    MultiHeadAttention._forward = attention_forward
    MultiHeadAttention._prepare_qkv_func = _prepare_qkv
    TransformerEncoderLayer._forward = transformer_encoder_layer_forward
    TransformerEncoder._forward = transformer_encoder_forward
    TinyBertForPretraining._forward = minilm_pretraining_forward

    def init_func(layer):
        if isinstance(layer, (MultiHeadAttention, TransformerEncoderLayer,
                              TransformerEncoder, TinyBertForPretraining)):
            layer.forward = layer._forward
            if isinstance(layer, MultiHeadAttention):
                layer._prepare_qkv = layer._prepare_qkv_func
                layer.num_relation_heads = num_relation_heads

    for layer in self.children():
        layer.apply(init_func)

    base_model_prefix = self._layers.base_model_prefix if isinstance(
        self, paddle.DataParallel) else self.base_model_prefix

    # For distribute training
    if isinstance(self, paddle.DataParallel):
        if hasattr(self._layers, base_model_prefix):
            self.outputs = getattr(self._layers, base_model_prefix).encoder
        else:
            self.outputs = self._layers.encoder
    else:
        if hasattr(self, base_model_prefix):
            self.outputs = getattr(self, base_model_prefix).encoder
        else:
            self.outputs = self.encoder
    return self


def _convert_attention_mask(attn_mask, dtype):
    if attn_mask is not None and attn_mask.dtype != dtype:
        attn_mask_dtype = convert_dtype(attn_mask.dtype)
        if attn_mask_dtype == 'bool' or 'int' in attn_mask_dtype:
            attn_mask = (paddle.cast(attn_mask, dtype) - 1.0) * 1e9
        else:
            attn_mask = paddle.cast(attn_mask, dtype)
    return attn_mask


def _prepare_qkv(self, query, key, value, cache=None):
    """
    Redefines the `_prepare_qkv` function of `paddle.nn.MultiHeadAttention`.
    While using MINILMv2's strategy and teacher's head num is not equal to
    student's head num, head num dim should be unified to a relation head num.
    """
    q = self.q_proj(query)
    if self.num_relation_heads != 0 and self.num_relation_heads != self.num_heads:
        assert self.num_heads * self.head_dim % self.num_relation_heads == 0, \
            "`num_relation_heads` must be divisible by `hidden_size`."
        self.head_dim = int(self.num_heads * self.head_dim /
                            self.num_relation_heads)
        self.num_heads = self.num_relation_heads
    q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
    q = tensor.transpose(x=q, perm=[0, 2, 1, 3])
    if isinstance(cache, self.StaticCache):
        # for encoder-decoder attention in inference and has cached
        k, v = cache.k, cache.v
    else:
        k, v = self.compute_kv(key, value)

    if isinstance(cache, self.Cache):
        # for decoder self-attention in inference
        k = tensor.concat([cache.k, k], axis=2)
        v = tensor.concat([cache.v, v], axis=2)
        cache = self.Cache(k, v)

    return (q, k, v) if cache is None else (q, k, v, cache)


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
    self.scaled_qk = product

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
    self.q = q
    self.k = k
    self.v = v

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
    self.scaled_qk = self.self_attn.scaled_qk
    self.q = self.self_attn.q
    self.k = self.self_attn.k
    self.v = self.self_attn.v
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
    self.scaled_qks = []
    self.qs, self.ks, self.vs = [], [], []
    for i, mod in enumerate(self.layers):
        self.hidden_states.append(output)
        if cache is None:
            output = mod(output, src_mask=src_mask)
        else:
            output, new_cache = mod(output, src_mask=src_mask, cache=cache[i])
            new_caches.append(new_cache)
        self.attentions.append(mod.attention_matrix)
        self.scaled_qks.append(mod.scaled_qk)
        self.qs.append(mod.q)
        self.ks.append(mod.k)
        self.vs.append(mod.v)

    if self.norm is not None:
        output = self.norm(output)
    self.hidden_states.append(output)
    return output if cache is None else (output, new_caches)


def minilm_pretraining_forward(self,
                               input_ids,
                               token_type_ids=None,
                               attention_mask=None):
    """
    Replaces `forward` function while using multi gpus to train. If training on
    single GPU, this `forward` could not be replaced.
    The type of `self` should inherit from base class of pretrained LMs, such as
    `TinyBertForPretraining`.
    Strategy MINILM only need q, k and v of transformers.
    """
    assert hasattr(self, self.base_model_prefix), \
        "Student class should inherit from %s" % (self.base_model_class)
    model = getattr(self, self.base_model_prefix)
    encoder = model.encoder

    sequence_output, pooled_output = model(input_ids, token_type_ids,
                                           attention_mask)
    return encoder.qs, encoder.ks, encoder.vs
