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
from paddle.fluid.data_feeder import convert_dtype
from paddle import tensor
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import MultiHeadAttention, TransformerEncoderLayer, TransformerEncoder

from paddlenlp.transformers.bert.modeling import BertPooler, BertEmbeddings

from .. import PretrainedModel, register_base_model

__all__ = [
    'TinyBertModel',
    'TinyBertPretrainedModel',
    'TinyBertForPretraining',
    'TinyBertForSequenceClassification',
    'to_distill',
]


def to_distill(self):
    """
    Can be bound to object with transformer layers, and exposes attributes
    `emb`, `outputs.hidden_states`and `outputs.attentions` of the object for
    distillation.
    """
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
    self.emb = getattr(self,
                       self.base_model_prefix).embeddings.word_embeddings.weight

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
    for i, mod in enumerate(self.layers):
        if cache is None:
            output = mod(output, src_mask=src_mask)
        else:
            output, new_cache = mod(output, src_mask=src_mask, cache=cache[i])
            new_caches.append(new_cache)
        self.attentions.append(mod.attention_matrix)
        self.hidden_states.append(output)

    if self.norm is not None:
        output = self.norm(output)
    self.hidden_states.append(output)

    return output if cache is None else (output, new_caches)


class TinyBertPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained TinyBERT models. It provides TinyBERT
    related `model_config_file`, `resource_files_names`,
    `pretrained_resource_files_map`, `pretrained_init_configuration`,
    `base_model_prefix` for downloading and loading pretrained models. See
    `PretrainedModel` for more details.
    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "tinybert-4l-312d": {
            "vocab_size": 30522,
            "hidden_size": 312,
            "num_hidden_layers": 4,
            "num_attention_heads": 12,
            "intermediate_size": 1200,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "tinybert-6l-768d": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 6,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "tinybert-4l-312d-v2": {
            "vocab_size": 30522,
            "hidden_size": 312,
            "num_hidden_layers": 4,
            "num_attention_heads": 12,
            "intermediate_size": 1200,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "tinybert-6l-768d-v2": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 6,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "tinybert-4l-312d":
            "http://paddlenlp.bj.bcebos.com/models/transformers/tinybert/tinybert-4l-312d.pdparams",
            "tinybert-6l-768d":
            "http://paddlenlp.bj.bcebos.com/models/transformers/tinybert/tinybert-6l-768d.pdparams",
            "tinybert-4l-312d-v2":
            "http://paddlenlp.bj.bcebos.com/models/transformers/tinybert/tinybert-4l-312d-v2.pdparams",
            "tinybert-6l-768d-v2":
            "http://paddlenlp.bj.bcebos.com/models/transformers/tinybert/tinybert-6l-768d-v2.pdparams",
        }
    }
    base_model_prefix = "tinybert"

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range") else
                        self.tinybert.config["initializer_range"],
                        shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


@register_base_model
class TinyBertModel(TinyBertPretrainedModel):
    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 pad_token_id=0):
        super(TinyBertModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.embeddings = BertEmbeddings(
            vocab_size, hidden_size, hidden_dropout_prob,
            max_position_embeddings, type_vocab_size)

        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        self.pooler = BertPooler(hidden_size)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id
                 ).astype(self.pooler.dense.weight.dtype) * -1e9,
                axis=[1, 2])
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layer = self.encoder(embedding_output, attention_mask)
        pooled_output = self.pooler(encoded_layer)

        return encoded_layer, pooled_output


class TinyBertForPretraining(TinyBertPretrainedModel):
    def __init__(self, tinybert, fit_size=768):
        super(TinyBertForPretraining, self).__init__()
        self.tinybert = tinybert
        self.fit_dense = nn.Linear(self.tinybert.config["hidden_size"],
                                   fit_size)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        sequence_output, pooled_output = self.tinybert(
            input_ids, token_type_ids, attention_mask)
        tmp = []
        for sequence_layer in sequence_output:
            tmp.append(self.fit_dense(sequence_layer))
        sequence_output = tmp

        return sequence_output


class TinyBertForSequenceClassification(TinyBertPretrainedModel):
    def __init__(self, tinybert, num_classes=2, dropout=None, fit_size=768):
        super(TinyBertForSequenceClassification, self).__init__()
        self.tinybert = tinybert
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.tinybert.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.tinybert.config["hidden_size"],
                                    num_classes)
        self.fit_dense = nn.Linear(self.tinybert.config["hidden_size"],
                                   fit_size)
        self.activation = nn.ReLU()
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        sequence_output, pooled_output = self.tinybert(
            input_ids, token_type_ids, attention_mask)

        logits = self.classifier(self.activation(pooled_output))
        return logits
