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
from paddlenlp.transformers import TinyBertForPretraining, TinyBertForSequenceClassification, BertForSequenceClassification

__all__ = ['to_distill', 'calc_minilm_loss']


def calc_minilm_loss(loss_fct, s, t, attn_mask, num_relation_heads=0):
    # Initialize head_num
    if num_relation_heads > 0 and num_relation_heads != s.shape[1]:
        # s'shape: [bs, seq_len, head_num, head_dim]
        s = tensor.transpose(x=s, perm=[0, 2, 1, 3])
        # s'shape: [bs, seq_len, num_relation_heads, head_dim_new]
        s = tensor.reshape(x=s, shape=[0, 0, num_relation_heads, -1])
        #s's shape: [bs, num_relation_heads, seq_len,, head_dim_new]
        s = tensor.transpose(x=s, perm=[0, 2, 1, 3])
    if num_relation_heads > 0 and num_relation_heads != t.shape[1]:
        t = tensor.transpose(x=t, perm=[0, 2, 1, 3])
        t = tensor.reshape(x=t, shape=[0, 0, num_relation_heads, -1])
        t = tensor.transpose(x=t, perm=[0, 2, 1, 3])

    pad_seq_len = s.shape[2]
    s_head_dim, t_head_dim = s.shape[3], t.shape[3]
    scaled_dot_product_s = tensor.matmul(
        x=s, y=s, transpose_y=True) / math.sqrt(s_head_dim)
    del s
    scaled_dot_product_s += attn_mask

    scaled_dot_product_t = tensor.matmul(
        x=t, y=t, transpose_y=True) / math.sqrt(t_head_dim)
    del t
    scaled_dot_product_t += attn_mask
    loss = loss_fct(
        F.log_softmax(scaled_dot_product_s), F.softmax(scaled_dot_product_t))
    return loss


def to_distill(self,
               return_qkv=False,
               return_attentions=False,
               return_layer_outputs=False,
               layer_index=-1):
    """
    Can be bound to object with transformer encoder layers, and make model
    expose attributes `outputs.qs`, `outputs.ks`, `outputs.vs`,
    `outputs.scaled_qks`, `outputs.hidden_states`and `outputs.attentions` of
    the object for distillation.
    """
    logger.warning("`to_distill` is an experimental API and subject to change.")
    MultiHeadAttention._forward = attention_forward
    TransformerEncoderLayer._forward = transformer_encoder_layer_forward
    TransformerEncoder._forward = transformer_encoder_forward
    BertForSequenceClassification._forward = bert_forward
    if return_qkv:
        TinyBertForPretraining._forward = minilm_pretraining_forward
    else:
        TinyBertForPretraining._forward = tinybert_forward

    def init_func(layer):
        if isinstance(layer, (MultiHeadAttention, TransformerEncoderLayer,
                              TransformerEncoder, TinyBertForPretraining,
                              BertForSequenceClassification)):
            layer.forward = layer._forward
            if isinstance(layer, TransformerEncoder):
                layer.return_layer_outputs = return_layer_outputs
                layer.layer_index = layer_index
            if isinstance(layer, MultiHeadAttention):
                layer.return_attentions = return_attentions
                layer.return_qkv = return_qkv

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

    self.attention_matrix = product if self.return_attentions else None
    weights = F.softmax(product)
    if self.dropout:
        weights = F.dropout(
            weights,
            self.dropout,
            training=self.training,
            mode="upscale_in_train")

    out = tensor.matmul(weights, v)
    if self.return_qkv:
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
    if hasattr(self.self_attn, 'attention_matrix'):
        self.attention_matrix = self.self_attn.attention_matrix
    if hasattr(self.self_attn, 'q'):
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

    for i, mod in enumerate(self.layers):
        if self.return_layer_outputs:
            self.hidden_states.append(output)
        if cache is None:
            output = mod(output, src_mask=src_mask)
        else:
            output, new_cache = mod(output, src_mask=src_mask, cache=cache[i])
            new_caches.append(new_cache)
        if hasattr(mod, 'attention_matrix'):
            self.attentions.append(mod.attention_matrix)
        if i == self.layer_index and hasattr(mod, 'q'):
            self.q = mod.q
            self.k = mod.k
            self.v = mod.v

    if self.norm is not None:
        output = self.norm(output)
    if self.return_layer_outputs:
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
    return encoder.q, encoder.k, encoder.v


def tinybert_forward(self, input_ids, token_type_ids=None, attention_mask=None):
    """
    Replaces `forward` function while using multi gpus to train.
    """
    assert hasattr(self, self.base_model_prefix), \
        "Student class should inherit from %s" % (self.base_model_class)
    model = getattr(self, self.base_model_prefix)
    encoder = model.encoder

    sequence_output, pooled_output = model(input_ids, token_type_ids,
                                           attention_mask)
    for i in range(len(encoder.hidden_states)):
        # While using tinybert-4l-312d, tinybert-6l-768d, tinybert-4l-312d-zh, tinybert-6l-768d-zh
        # While using tinybert-4l-312d-v2, tinybert-6l-768d-v2
        # encoder.hidden_states[i] = self.tinybert.fit_dense(encoder.hidden_states[i])
        encoder.hidden_states[i] = self.tinybert.fit_denses[i](
            encoder.hidden_states[i])

    return encoder.attentions, encoder.hidden_states


def bert_forward(self, input_ids, token_type_ids=None, attention_mask=None):
    """
    Replaces `forward` function while using multi gpus to train.
    """
    assert hasattr(self, self.base_model_prefix), \
        "Student class should inherit from %s" % (self.base_model_class)
    model = getattr(self, self.base_model_prefix)
    encoder = model.encoder

    sequence_output, pooled_output = model(input_ids, token_type_ids,
                                           attention_mask)
    return encoder.attentions, encoder.hidden_states
